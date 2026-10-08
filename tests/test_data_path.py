"""Data-path regression tests and reproducible pilot audit (stdlib unittest, CPU only).

PYTHONPATH=. python tests/test_data_path.py
PYTHONPATH=. python tests/test_data_path.py --audit-pool data/prior/pilot_graph --output graph_audit.json
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lightpfn.prior import generate as gen
from lightpfn.train import PoolStream, _check_loaded_group, inject_missing, micro_batches

TMP = Path(os.environ.get('LIGHTPFN_TEST_TMP', tempfile.gettempdir()))


def numpy_transport(group):
    # The audit sandbox forbids AF_UNIX sockets used by torch's tensor resource_sharer.
    # Pickle NumPy arrays across the worker pipe; PoolStream itself still emits tensors.
    return {k:(v.numpy() if torch.is_tensor(v) else v) for k,v in group.items()}


def simple_group(B=8,n=64,m=5):
    rng=np.random.default_rng(22)
    X=rng.normal(size=(B,n,m)).astype(np.float16)
    d=np.array([i%m+1 for i in range(B)],dtype=np.int16)
    for b,db in enumerate(d):X[b,:,db:]=0
    y=np.tile(np.arange(n)%2,(B,1)).astype(np.uint8)
    return dict(X=torch.from_numpy(X),y=torch.from_numpy(y),d=torch.from_numpy(d),
                n_classes=torch.full((B,),2,dtype=torch.int16),n_train=32,
                resampled=torch.zeros(B,dtype=torch.bool))


class RecordingRng:
    def __init__(self,seed,mech):
        self.rng=np.random.default_rng(seed);self.mech=mech;self.rates=[];self.cols=[];self.source_bounds=[]
    def uniform(self,low,high):
        v=self.rng.uniform(low,high)
        if low<0:self.rates.append(np.exp(v))
        return v
    def choice(self,a,*args,**kwargs):
        if isinstance(a,list) and a==['mcar','mar','mnar']:return self.mech
        v=self.rng.choice(a,*args,**kwargs)
        if isinstance(a,int) and 'size' in kwargs:self.cols=v
        return v
    def integers(self,high):self.source_bounds.append(high);return self.rng.integers(high)
    def random(self,n):return self.rng.random(n)


def missing_rate_measurement():
    results={}
    for mech in ('mcar','mar','mnar'):
        observed=[];expected=[]
        for i in range(150):
            # Ties are intentional: ranks must not introduce an artificial row-order trend.
            X=np.random.default_rng(i).integers(0,8,size=(1200,6)).astype(float)
            rng=RecordingRng(500+i,mech)
            inject_missing(rng,X,6,700)
            observed.append(float(np.isnan(X).mean()))
            expected.append(float(sum(rng.rates)/6))
            assert np.isfinite(X[:700]).sum(0).min()>=2
            if mech=='mar':assert rng.source_bounds and set(rng.source_bounds)=={5}
        results[mech]=dict(observed=float(np.mean(observed)),expected=float(np.mean(expected)),tables=len(observed))
    return results


class DataPathTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=TMP)
        self.path=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_missingness_rates_and_mar_source(self):
        for result in missing_rate_measurement().values():
            self.assertLess(abs(result['observed']-result['expected']),.002)

    def test_missingness_small_train_width_padding_and_determinism(self):
        for n,ntr in ((3,1),(4,2),(32,3),(100,99)):
            for d in (1,2,5):
                for seed in range(100):
                    X=np.random.default_rng(1).normal(size=(n,d+2));X[:,d:]=0
                    a,b=X.copy(),X.copy()
                    inject_missing(np.random.default_rng(seed),a,d,ntr)
                    inject_missing(np.random.default_rng(seed),b,d,ntr)
                    np.testing.assert_array_equal(a,b)
                    self.assertTrue(np.all(np.isfinite(a[:ntr,:d]).sum(0)>=min(2,ntr)))
                    self.assertTrue(np.all(a[:,d:]==0))
        X=np.ones((5,2));X[:3,0]=np.nan
        with self.assertRaises(ValueError):inject_missing(np.random.default_rng(0),X,2,3)
        for d,ntr in ((0,3),(3,3),(1,0),(1,6)):
            with self.assertRaises(ValueError):inject_missing(np.random.default_rng(0),np.ones((5,2)),d,ntr)

    def test_mar_does_not_propagate_new_nan_masks(self):
        # Compare against the same immutable source using an RNG that reverses selected columns.
        class Controlled:
            def __init__(self,reverse=False):self.reverse=reverse
            def uniform(self,low,high):return .99 if low>0 else np.log(.5)
            def choice(self,a,*args,**kwargs):
                if isinstance(a,list) and isinstance(a[0],str):return 'mar'
                if isinstance(a,list):return 1
                return np.array([1,0]) if self.reverse else np.array([0,1])
            def integers(self,high):return 0
            def random(self,n):return np.full(n,.4)
        X=np.stack([np.arange(100),np.arange(100)[::-1]],1).astype(float)
        a,b=X.copy(),X.copy()
        inject_missing(Controlled(),a,2,70);inject_missing(Controlled(True),b,2,70)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(np.isnan(a[:,0]),np.isnan(a[:,1])))

    def test_feature_permutation_preserves_columns_and_input(self):
        g=simple_group();original=copy.deepcopy(g)
        out=PoolStream([(self.path,1)],p_missing=0).augment(np.random.default_rng(0),g)
        for b,db in enumerate(g['d']):
            db=int(db)
            np.testing.assert_array_equal(np.unique(out['X'][b,:,:db].T.numpy(),axis=0),
                                           np.unique(g['X'][b,:,:db].float().T.numpy(),axis=0))
            self.assertTrue(torch.all(out['X'][b,:,db:]==0))
        for key in g:
            if torch.is_tensor(g[key]):torch.testing.assert_close(g[key],original[key],rtol=0,atol=0)
        self.assertEqual(out['y'].dtype,torch.int64)

    def test_stream_determinism_and_missingness(self):
        torch.save([simple_group()],self.path/'shard_000000.pt')
        a=iter(PoolStream([(self.path,1)],p_missing=1,seed=8))
        b=iter(PoolStream([(self.path,1)],p_missing=1,seed=8))
        for _ in range(12):
            ga,gb=next(a),next(b)
            for k in ga:
                if torch.is_tensor(ga[k]):torch.testing.assert_close(ga[k],gb[k],rtol=0,atol=0,equal_nan=True)
                else:self.assertEqual(ga[k],gb[k])
            for i,d in enumerate(ga['d']):
                self.assertTrue(torch.isfinite(ga['X'][i,:ga['n_train'],:int(d)]).sum(0).min()>=2)
                self.assertTrue(torch.all(ga['X'][i,:,int(d):]==0))

    def test_dataloader_worker_streams_are_reproducible_and_distinct(self):
        torch.save([simple_group()],self.path/'shard_000000.pt')
        def read():
            loader=torch.utils.data.DataLoader(PoolStream([(self.path,1)],p_missing=1,seed=99),
                                              batch_size=None,num_workers=2,collate_fn=numpy_transport)
            it=iter(loader)
            try:return [next(it)['X'] for _ in range(8)]
            finally:it._shutdown_workers()
        a,b=read(),read()
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertFalse(np.array_equal(np.isnan(a[0]),np.isnan(a[1])))

    def test_pool_weights_apply_per_group(self):
        second=self.path/'second';second.mkdir()
        a,b=simple_group(B=1),simple_group(B=1)
        a['X']*=0.1;b['X']*=2
        torch.save([a]*1,self.path/'shard_000000.pt')
        torch.save([b]*9,second/'shard_000000.pt')
        stream=iter(PoolStream([(self.path,.75),(second,.25)],p_missing=0,seed=67))
        high=sum(float(next(stream)['X'].abs().max())>1 for _ in range(3000))
        self.assertLess(abs(high/3000-.25),.03)

    def test_empty_corrupt_pools_and_invalid_inputs(self):
        with self.assertRaises(FileNotFoundError):next(iter(PoolStream([(self.path,1)])))
        (self.path/'shard_000000.pt.tmp').write_bytes(b'partial')
        with self.assertRaises(FileNotFoundError):next(iter(PoolStream([(self.path,1)])))
        (self.path/'shard_000000.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'shard_000000'):next(iter(PoolStream([(self.path,1)])))
        for pools in ([],[(self.path,0)],[(self.path,-1)],[(self.path,float('nan'))]):
            with self.assertRaises(ValueError):PoolStream(pools)
        for p in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):PoolStream([(self.path,1)],p_missing=p)
        for seed in (-1,2**32,.5):
            with self.assertRaises(ValueError):PoolStream([(self.path,1)],seed=seed)

    def test_large_weights_normalize_without_overflow(self):
        torch.save([simple_group()],self.path/'shard_000000.pt')
        g=next(iter(PoolStream([(self.path,1e308),(self.path,1e308)],p_missing=0)))
        self.assertTrue(torch.isfinite(g['X']).all())

    def test_loaded_labels_and_float_range(self):
        g=simple_group()
        for change in (lambda g:g['n_classes'].__setitem__(0,17),
                       lambda g:g['y'].__setitem__((0,0),255),
                       lambda g:g['y'].__setitem__((0,slice(None,32)),0),
                       lambda g:g['X'].__setitem__((0,0,0),float('nan')),
                       lambda g:g['X'].__setitem__((0,0,0),1001),
                       lambda g:g['X'].__setitem__((0,0,1),1),
                       lambda g:g['d'].__setitem__(0,0)):
            bad=copy.deepcopy(g);change(bad)
            with self.assertRaises(ValueError):_check_loaded_group(bad)

    def test_legacy_groups_without_test_class_or_with_constant_column_load(self):
        # v1 pools: a rare class can be missing from a small test split, and a column can be
        # constant after the group cut; training only needs every class in the train rows
        for change in (lambda g:g['y'].__setitem__((0,slice(32,None)),0),
                       lambda g:g['X'].__setitem__((0,slice(None),0),1)):
            g=simple_group();change(g)
            _check_loaded_group(g)

    def test_duplicate_rows_columns_and_constant_train_are_supported(self):
        g=simple_group(B=2,m=5)
        g['d'].fill_(2);g['X'][:,:,2:]=0
        g['X'][:,:32,:2]=0
        g['X'][:,32:,:2]=1
        _check_loaded_group(g)
        gen.validate_group(g)
        torch.save([g],self.path/'shard_000000.pt')
        out=next(iter(PoolStream([(self.path,1)],p_missing=1,seed=71)))
        self.assertEqual(out['X'].shape,(2,64,5))
        self.assertTrue(torch.all(out['X'][:,:,2:]==0))
        self.assertTrue(torch.isfinite(out['X'][:,:32,:2]).sum(1).min()>=2)
        mb=list(micro_batches(out,100000))[0]
        self.assertEqual(mb['X'].shape,(2,64,2))
        torch.testing.assert_close(out['y'],g['y'].long())

    def test_zero_weight_pool_is_ignored_and_mix_after_missingness(self):
        absent=self.path/'absent'
        torch.save([simple_group()],self.path/'shard_000000.pt')
        stream=iter(PoolStream([(absent,0),(self.path,1)],p_missing=1))
        for _ in range(3):
            g=next(stream)
            self.assertTrue(torch.isnan(g['X']).any())
            self.assertTrue(torch.equal(torch.unique(g['y']),torch.arange(2)))

    def test_micro_batches_preserve_all_tasks_soft_budget_and_trim(self):
        g=PoolStream([(self.path,1)],p_missing=0).augment(np.random.default_rng(0),simple_group())
        g['scalar']=torch.tensor(3)
        for budget in (1,320,640,1000,100000):
            batches=list(micro_batches(g,budget))
            self.assertEqual(sum(b['X'].shape[0] for b in batches),8)
            torch.testing.assert_close(torch.cat([b['y'] for b in batches]),g['y'])
            for b in batches:
                self.assertEqual(b['X'].shape[-1],int(b['d'].max()))
                self.assertTrue(b['X'].numel()<=budget or b['X'].shape[0]==1)
                self.assertEqual(b['scalar'].item(),3)
        with self.assertRaises(ValueError):list(micro_batches(g,0))

    def test_micro_batches_row_cost_and_fit_rows(self):
        # rows carry their index in column 0 and the train/test part in column 1, so alignment can be checked
        n,ntr,m=1000,600,5
        X=torch.zeros(2,n,m);X[:,:,0]=torch.arange(n,dtype=torch.float32);X[:,ntr:,1]=1
        y=torch.zeros(2,n,dtype=torch.long);y[:,::3]=1;y[0,:10]=2;y[1,:ntr:7]=2  # class 2 rare in task 0
        g=dict(X=X,y=y,d=torch.tensor([5,3]),n_classes=torch.tensor([3,3]),n_train=ntr)
        self.assertEqual(len(list(micro_batches(g,2*n*m))),1)
        self.assertEqual(len(list(micro_batches(g,2*n*m,row_cost=3))),2)  # cost n*(m+3) per task
        self.assertEqual(list(micro_batches(g,n*m,row_cost=3))[0]['X'].shape,(1,n,5))  # no rng: over budget
        mbs=list(micro_batches(g,400*(m+3),row_cost=3,rng=np.random.default_rng(0)))
        self.assertEqual([mb['X'].shape for mb in mbs],[(1,400,5),(1,400,3)])
        for t,mb in enumerate(mbs):
            self.assertEqual(mb['n_train'],240)
            idx=mb['X'][0,:,0].long()
            self.assertEqual(len(set(idx.tolist())),400)
            self.assertTrue(bool((idx[:240]<ntr).all()) and bool((idx[240:]>=ntr).all()))
            torch.testing.assert_close(mb['y'][0],y[t,idx])
            self.assertEqual(set(mb['y'][0,:240].tolist()),{0,1,2})
            self.assertEqual(mb['X'].shape[-1],int(mb['d'].max()))
        # padding columns cost nothing: 2 real features of 13 stored, n*(2+3) fits the budget exactly
        gp=dict(X=torch.zeros(2,n,13),y=y,d=torch.tensor([2,2]),n_classes=torch.tensor([3,3]),n_train=ntr)
        mbs=list(micro_batches(gp,n*(2+3),row_cost=3,rng=np.random.default_rng(0)))
        self.assertEqual([mb['X'].shape for mb in mbs],[(1,n,2),(1,n,2)])

    def test_v2_augmentation_determinism_padding_and_source_immutability(self):
        g = simple_group(B=32, n=64, m=12)
        original = copy.deepcopy(g)
        stream = PoolStream([(self.path, 1)], p_missing=1, seed=7, augment='v2')
        changed = grew = False
        for seed in range(40):
            a = stream.augment(np.random.default_rng(seed), g)
            b = stream.augment(np.random.default_rng(seed), g)
            torch.testing.assert_close(a['X'], b['X'], rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(a['d'], b['d'], rtol=0, atol=0)
            changed |= bool(torch.isnan(a['X']).any())
            grew |= bool((a['d'] > g['d']).any())
            for i, d in enumerate(a['d']):
                self.assertLessEqual(int(d), 12)
                self.assertTrue(torch.all(a['X'][i, :, int(d):] == 0))
                finite = a['X'][i, :, :int(d)]
                self.assertTrue(torch.all(finite[torch.isfinite(finite)].abs() <= 1000))
                self.assertTrue(torch.isfinite(finite[:32]).sum(0).min() >= 2)
            batches = list(micro_batches(a, 1000))
            self.assertEqual(sum(len(mb['d']) for mb in batches), 32)
            for mb in batches:
                self.assertEqual(mb['cat_mask'].shape, (len(mb['d']), mb['X'].shape[-1]))
                for i, width in enumerate(mb['d']):
                    self.assertFalse(mb['cat_mask'][i, int(width):].any())
        self.assertTrue(changed and grew)
        for key in g:
            if torch.is_tensor(g[key]): torch.testing.assert_close(g[key], original[key], rtol=0, atol=0)
        with self.assertRaises(ValueError): PoolStream([(self.path, 1)], augment='invalid')

    def test_v2_monotonicity_quantization_and_full_width(self):
        from lightpfn.train import augment_v2
        class Controlled:
            def __init__(self, kind): self.rng=np.random.default_rng(4); self.kind=kind; self.calls=0
            def __getattr__(self, key): return getattr(self.rng, key)
            def random(self):
                self.calls += 1
                return 0.0 if self.calls == self.kind else 1.0
        X = np.tile(np.linspace(-1000, 1000, 200)[:, None], (1, 2)).astype(np.float32)
        for kind in (1, 2, 3):
            a = X.copy(); cat = np.zeros(2, dtype=bool)
            d = augment_v2(Controlled(kind), a, 2, 120, cat)
            self.assertEqual(d, 2); self.assertTrue(np.isfinite(a).all())
            if kind == 1:
                self.assertTrue(np.all(np.diff(a, axis=0) >= 0))
            if kind == 2:
                self.assertTrue(cat.any())
                self.assertTrue(any(np.any(np.diff(a[:, j]) < 0) for j in np.flatnonzero(cat)))
        a = np.ones((20, 2), dtype=np.float32)
        augment_v2(np.random.default_rng(0), a, 1, 10, np.zeros(2, dtype=bool))
        self.assertTrue(np.isfinite(a).all())
        a = X.copy(); cat = np.zeros(2, dtype=bool)
        augment_v2(Controlled(2), a, 2, 1, cat)
        self.assertFalse(cat.any())  # one train row cannot define quantile bins
        for x, d, mask in ((X.astype(int), 2, np.zeros(2, dtype=bool)),
                           (X, 1.5, np.zeros(2, dtype=bool)),
                           (X, 2, np.zeros(2)),
                           (X*2, 2, np.zeros(2, dtype=bool))):
            with self.assertRaises(ValueError): augment_v2(np.random.default_rng(0), x.copy(), d, 120, mask)

    def test_dev_probes_compatibility_and_metrics(self):
        from lightpfn.eval import dev, probes
        self.assertEqual(len(probes.PROBES), 25)
        for name, d in probes.PROBES:
            X, y = probes.probe(name, d, np.random.default_rng(5))
            self.assertEqual(X.shape, (probes.N_TRAIN + probes.N_TEST, d))
            self.assertEqual(X.dtype, np.float32)
            self.assertTrue(np.isfinite(X).all())
            np.testing.assert_array_equal(np.unique(y), [0, 1])
        result = dev.metrics(np.array([0, 1, 0, 1]), np.array([[.9,.1],[.1,.9],[.8,.2],[.2,.8]]))
        self.assertEqual(result['auc'], 1)
        self.assertGreater(result['logloss'], 0)
        for count, jobs in ((7, 1), (8, 13), (0, 1)):
            with self.assertRaises(ValueError): dev.build_dev(self.path, count, n_jobs=jobs)

    def test_dev_set_task_ids_evaluation_and_corruption(self):
        from argparse import Namespace
        import json
        from unittest.mock import patch
        from lightpfn.eval import dev
        from lightpfn.eval.probes import probe
        base = self.path/'dev_fixture'
        for family, (prior, version, pbin) in dev.FAMILIES.items():
            folder=base/family; folder.mkdir(parents=True)
            args=Namespace(preset='s1b', prior=prior, prior_version=version, geometry='ratio',
                p_binary=pbin, out=str(folder), n_tasks=8, n_jobs=1, max_features=4,
                groups_per_shard=1, seed=dev.DEV_SEED, min_free_gb=0.)
            class Serial:
                def map(self, fn, specs, chunksize): return [fn(s) for s in specs]
            groups=gen.make_shard(Serial(), args, 0, 1)
            (folder/'meta.json').write_text(json.dumps(gen.generator_meta(args)))
            gen.atomic_write(folder/'shard_000000.pt', lambda f: torch.save(groups, f))
        rows=list(dev.dev_tasks(base, limit=2))
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({(r[0], r[1]) for r in rows}), 8)
        class Uniform:
            def fit(self, X, y): self.C=len(np.unique(y)); return self
            def predict_proba(self, X): return np.full((len(X),self.C), 1/self.C)
        # Redirect eval outputs into the temporary repo folder, keep the real data dir.
        with patch.object(dev, 'ROOT', self.path):
            a,b=dev.evaluate_checkpoint('unused', 'fixture', dev_dir=base, device='cpu',
                threads=1, limit=1, probe_limit=1, probe_seeds=1, classifier=Uniform())
        self.assertEqual(len(a),4); self.assertEqual(len(b),1)
        self.assertTrue(np.isfinite(a[['auc','logloss']]).all().all())
        self.assertEqual(set(a.family),set(dev.FAMILIES))
        self.assertTrue((self.path/'runs/dev/fixture/D1.csv').exists())
        folder=base/'rule'; meta=json.loads((folder/'meta.json').read_text())
        meta['n_tasks']=16; (folder/'meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            list(dev.dev_tasks(base, limit=1))

    def test_model_cpu_supported_edge_inputs(self):
        from lightpfn.model.lightpfn import Config,LightPFN
        torch.manual_seed(9);model=LightPFN(Config()).eval()
        for p in model.parameters():torch.nn.init.normal_(p,std=.03)
        y=torch.arange(24)[None]%2
        for d in (1,2,5):
            for kind in ('all_nan_train','constant','max_pool_value','float16_max'):
                X=torch.randn(1,32,d)
                if kind=='all_nan_train':X[:,:24,0]=float('nan')
                elif kind=='constant':X[:,:,0]=7
                elif kind=='max_pool_value':X*=1000
                else:X*=65504
                with torch.no_grad():out=model(X,y,n_classes=2)
                self.assertEqual(out.shape,(1,8,2));self.assertTrue(torch.isfinite(out).all())
        # Maximum prior class count, smallest width, and arbitrary model label slots.
        X=torch.randn(1,32,1);y=torch.arange(24)[None]%10
        slots=torch.randperm(16)[None]
        with torch.no_grad():out=model(X,y,slots=slots,n_classes=10)
        self.assertEqual(out.shape,(1,8,10));self.assertTrue(torch.isfinite(out).all())

    def test_harness_checkpoint_cli_factory(self):
        from unittest.mock import patch
        from lightpfn.eval import harness
        from test_release import small_model

        calls = []
        def evaluate(model, jobs, threads, workers, out):
            calls.append((model, threads, workers, out))
            if callable(model):
                clf = model(threads, 7)
                self.assertEqual((clf.device, clf.n_estimators, clf.n_threads, clf.seed), ("cpu", 2, 3, 7))
                clf.fit(np.arange(24).reshape(8, 3), np.arange(8) % 2)
                self.assertEqual(clf.predict_proba([[1, 2, 3]]).shape, (1, 2))

        argv = ["harness", "--models", "catboost", "--checkpoint", "weights", "--name", "local",
                "--n-estimators", "2", "--device", "cpu", "--n-threads", "3", "--n-workers", "1"]
        with patch.object(sys, "argv", argv), patch.object(harness, "RUNS_DIR", self.path / "eval"), \
             patch.object(harness, "load_tasks", return_value=[]), patch.object(harness, "make_jobs", return_value=[]), \
             patch.object(harness, "evaluate", side_effect=evaluate), \
             patch("lightpfn.sklearn.load_model", return_value=small_model()) as load:
            harness.main()
        load.assert_called_once_with("weights", "cpu")
        self.assertEqual([c[3].name for c in calls], ["catboost.csv", "local.csv"])

    def test_harness_invalid_cli_fails_before_loading_data(self):
        import contextlib
        import io
        from unittest.mock import patch
        from lightpfn.eval import harness

        for args in ([], ["--checkpoint", "missing"], ["--name", "orphan"],
                     ["--models", "catboost", "--n-workers", "0"],
                     ["--checkpoint", "missing", "--name", "local", "--n-estimators", "0"]):
            with patch.object(sys, "argv", ["harness", *args]), patch.object(harness, "load_tasks") as load, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                harness.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()


def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(min=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),
                p90=float(np.quantile(a,.9)),max=float(a.max()),mean=float(a.mean()))


def filter_pvalue(X,y,seed=1):
    from sklearn.ensemble import ExtraTreesRegressor
    C=int(np.max(y))+1
    Y=np.eye(C,dtype=np.float32)[y]
    if C==2:Y=Y[:,:1]
    et=ExtraTreesRegressor(n_estimators=25,bootstrap=True,oob_score=True,n_jobs=1,random_state=seed,max_depth=6)
    et.fit(X,Y[:,0] if C==2 else Y)
    P=et.oob_prediction_.reshape(len(y),-1)
    imp=((Y-Y.mean(0))**2).sum(1)-((Y-P)**2).sum(1)
    idx=np.random.default_rng(0).integers(0,len(y),(200,len(y)))
    return float(np.mean(imp[idx].mean(1)<=0))


def audit_pool(path):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score
    path=Path(path);meta=json.loads((path/'meta.json').read_text())
    lengths=gen.check_resume(path,meta)
    metrics=[];rejected=invalid=drafts=checks=requested=resampled=0
    source_tasks=[]
    for si,file in enumerate(sorted(path.glob('shard_*.pt'))):
        groups=torch.load(file,map_location='cpu',weights_only=True)
        for gi,g in enumerate(groups):
            gen.validate_group(g,preset=meta['preset_cfg'],max_features=meta['max_features'],
                               group_size=meta['preset_cfg']['group_size'])
            _check_loaded_group(g)
            for key in ('rejected','invalid','drafts','filter_checks','resample_requested','resampled'):
                value=int(g[key].sum())
                if key=='rejected':rejected+=value
                elif key=='invalid':invalid+=value
                elif key=='drafts':drafts+=value
                elif key=='filter_checks':checks+=value
                elif key=='resample_requested':requested+=value
                else:resampled+=value
            for bi in range(len(g['d'])):
                n,d,C,ntr=g['X'].shape[1],int(g['d'][bi]),int(g['n_classes'][bi]),g['n_train']
                X=g['X'][bi,:,:d].float().numpy();y=g['y'][bi].numpy().astype(int)
                model=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=2,max_features=1.,n_jobs=1,random_state=123)
                model.fit(X[:ntr],y[:ntr]);P=model.predict_proba(X[ntr:])
                auc=roc_auc_score(y[ntr:],P[:,1]) if C==2 else roc_auc_score(y[ntr:],P,multi_class='ovr',average='macro')
                final_filter=bool(gen._unpredictable(X,y))
                counts=np.bincount(y,minlength=C)
                d_requested=g['X'].shape[-1]
                entry=dict(shard=si,group=gi,task=bi,n=n,d=d,d_requested=d_requested,
                    d_fraction=d/d_requested,d_removed=d_requested-d,
                    ratio=d/n,requested_ratio=d_requested/n,C=C,n_train=ntr,
                    duplicate_rows=n-len(np.unique(X,axis=0)),
                    duplicate_columns=d-len(np.unique(X.T,axis=0)),
                    minority=float(counts.min()/n),imbalance=float(counts.max()/counts.min()),
                    resampled=bool(g['resampled'][bi]),requested=bool(g['resample_requested'][bi]),
                    auc=float(auc),final_filter=final_filter,min_train_class=int(np.bincount(y[:ntr],minlength=C).min()),
                    min_test_class=int(np.bincount(y[ntr:],minlength=C).min()))
                metrics.append(entry)
                # Repeat the identical array, vary ET randomness and vary row ordering, separately.
                if len(source_tasks)<32:
                    fixed=[gen._unpredictable(X,y) for _ in range(3)]
                    varied=[filter_pvalue(X,y,seed=s)>=.05 for s in range(1,9)]
                    permuted=[]
                    for s in range(8):
                        perm=np.random.default_rng(s).permutation(n)
                        permuted.append(gen._unpredictable(X[perm],y[perm]))
                    source_tasks.append(dict(**entry,fixed=fixed,varied=varied,permuted=permuted))
    # Every committed group traverses the actual augment -> microbatch path, including p_missing=1.
    stream=PoolStream([(path,1)],p_missing=1,seed=22)
    total_micro=0
    for file in sorted(path.glob('shard_*.pt')):
        for g in torch.load(file,map_location='cpu',weights_only=True):
            out=stream.augment(np.random.default_rng(100+total_micro),g)
            for b,d in enumerate(out['d']):
                assert torch.isfinite(out['X'][b,:out['n_train'],:int(d)]).sum(0).min()>=2
                assert torch.all(out['X'][b,:,int(d):]==0)
            mb=list(micro_batches(out,600000))
            assert sum(x['X'].shape[0] for x in mb)==len(g['d'])
            total_micro+=len(mb)
    # Exercise file selection, cache, shuffle and actual iterator as well.
    it=iter(stream)
    for _ in range(2*sum(lengths)):
        g=next(it)
        assert len(list(micro_batches(g,600000)))>=1
    summary=dict(tasks=len(metrics),groups=sum(lengths),shards=len(lengths),
                 binary_fraction=float(np.mean([t['C']==2 for t in metrics])),
                 class_histogram={str(c):sum(t['C']==c for t in metrics) for c in range(2,11)},
                 resample_requested=requested/len(metrics),resampled=resampled/len(metrics),
                 resample_success_given_requested=resampled/requested if requested else None,
                 filter_rejected=rejected,filter_checks=checks,filter_rejection_rate=rejected/checks,
                 structural_rejected=invalid,drafts=drafts,structural_rejection_rate=invalid/drafts,
                 trivial_auc_ge_098=float(np.mean([t['auc']>=.98 for t in metrics])),
                 auc_eq_1=float(np.mean([t['auc']==1 for t in metrics])),
                 minority_lt_010=float(np.mean([t['minority']<.1 for t in metrics])),
                 binary_minority_lt_010=float(np.mean([t['minority']<.1 for t in metrics if t['C']==2])),
                 width_reduced=float(np.mean([t['d']<t['d_requested'] for t in metrics])),
                 tasks_with_duplicate_rows=sum(t['duplicate_rows']>0 for t in metrics),
                 tasks_with_duplicate_columns=sum(t['duplicate_columns']>0 for t in metrics),
                 no_signal_auc_045_055=float(np.mean([.45<=t['auc']<=.55 for t in metrics])),
                 auc_lt_045=float(np.mean([t['auc']<.45 for t in metrics])),microbatches=total_micro)
    for key in ('n','d','d_requested','d_fraction','d_removed','ratio','requested_ratio',
                'C','n_train','minority','imbalance','auc'):
        summary[key]=describe([t[key] for t in metrics])
    for label,selected in [('all',metrics),('resampled',[t for t in metrics if t['resampled']]),
                           ('not_resampled',[t for t in metrics if not t['resampled']]),
                           ('minority_lt_005',[t for t in metrics if t['minority']<.05]),
                           ('final_filter_rejected',[t for t in metrics if t['final_filter']])]:
        summary['audit_'+label]=dict(tasks=len(selected),final_filter_rejection_rate=float(np.mean([t['final_filter'] for t in selected])) if selected else None,
                                   mean_auc=float(np.mean([t['auc'] for t in selected])) if selected else None)
    summary['width_fraction_histogram']={
        'eq_1':sum(t['d_fraction']==1 for t in metrics),
        '0.9_to_lt_1':sum(.9<=t['d_fraction']<1 for t in metrics),
        '0.75_to_lt_0.9':sum(.75<=t['d_fraction']<.9 for t in metrics),
        '0.5_to_lt_0.75':sum(.5<=t['d_fraction']<.75 for t in metrics),
        'lt_0.5':sum(t['d_fraction']<.5 for t in metrics)}
    summary['filter_stability']=dict(tasks=len(source_tasks),
        identical_array_flips=sum(len(set(t['fixed']))>1 for t in source_tasks),
        varied_seed_flips=sum(len(set(t['varied']))>1 for t in source_tasks),
        row_permutation_flips=sum(len(set(t['permuted']))>1 for t in source_tasks))
    from lightpfn.model.lightpfn import Config,LightPFN
    torch.manual_seed(19);model=LightPFN(Config()).eval()
    for p in model.parameters():torch.nn.init.normal_(p,std=.03)
    cpu_checks=[]
    selections=[max(metrics,key=lambda t:t['n']),max(metrics,key=lambda t:t['d']),
                min(metrics,key=lambda t:t['n']),min(metrics,key=lambda t:t['minority'])]
    for t in selections:
        group=torch.load(path/f"shard_{t['shard']:06d}.pt",weights_only=True)[t['group']]
        X=group['X'][t['task']:t['task']+1,:,:t['d']].float()
        inject_missing(np.random.default_rng(44),X[0].numpy(),t['d'],t['n_train'])
        with torch.no_grad():out=model(X,group['y'][t['task']:t['task']+1,:t['n_train']].long(),n_classes=t['C'])
        assert out.shape==(1,t['n']-t['n_train'],t['C']) and torch.isfinite(out).all()
        cpu_checks.append(dict(shard=t['shard'],group=t['group'],task=t['task'],n=t['n'],d=t['d'],C=t['C'],finite=True))
    return dict(meta=meta,summary=summary,tasks=metrics,filter_stability_tasks=source_tasks,
                missingness_rates=missing_rate_measurement(),model_cpu_checks=cpu_checks)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if '--audit-pool' in sys.argv:
            parser=argparse.ArgumentParser();parser.add_argument('--audit-pool',required=True);parser.add_argument('--output',required=True)
            args=parser.parse_args();result=audit_pool(args.audit_pool)
            Path(args.output).write_text(json.dumps(result,indent=2))
            print(json.dumps(result['summary'],indent=2))
        else:unittest.main(verbosity=2)
