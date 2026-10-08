"""Prior/shard contract tests, no pytest required: PYTHONPATH=. python tests/test_prior.py."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lightpfn.prior import generate as gen
from lightpfn.prior import trees

TMP = Path(os.environ.get('LIGHTPFN_TEST_TMP', tempfile.gettempdir()))


def fake_dataset(rng, spec, n):
    X = rng.normal(size=(n, spec['n_features'])).astype(np.float32)
    y = np.arange(n) % spec['n_classes']
    return X, y


def spec(n=256, d=4, C=2, prior='tree', seed=0):
    return dict(n_rows=n, n_features=d, n_classes=C, prior=prior, seed=seed)


def fake_task(seed=0, **kw):
    with patch.object(gen, '_base_dataset', fake_dataset), patch.object(gen, '_unpredictable', return_value=False):
        return gen.generate_task(spec(seed=seed, **kw))


class PriorTests(unittest.TestCase):
    def test_seventh_draft_is_checked_and_exhaustion_raises(self):
        with patch.object(gen, '_base_dataset', side_effect=fake_dataset) as base, \
                patch.object(gen, '_unpredictable', return_value=True) as check, \
                patch.object(gen, 'FILTER_TRIES', 6):
            with self.assertRaisesRegex(RuntimeError, 'predictability retries exhausted'):
                gen.generate_task(spec())
            self.assertEqual(base.call_count, 7)
            self.assertEqual(check.call_count, 7)

    def test_success_after_rejections_and_no_extra_filter_on_imbalance(self):
        with patch.object(gen, '_base_dataset', fake_dataset), \
                patch.object(gen, '_unpredictable', side_effect=[True]*6+[False]) as check:
            t = gen.generate_task(spec())
        self.assertEqual(t['rejected'], 6)
        self.assertEqual(t['filter_checks'], 7)
        self.assertEqual(check.call_count, 7)

    def test_resampling_exact_rows_extreme_proportions(self):
        for C in (2, 3, 10):
            with patch.object(gen, '_base_dataset', fake_dataset), \
                    patch.object(gen, '_unpredictable', return_value=False), \
                    patch.object(gen, 'P_RESAMPLE', 1), \
                    patch.object(gen, '_target_proportions', return_value=np.r_[1e-50, np.ones(C-1)]):
                t = gen.generate_task(spec(C=C))
            self.assertEqual(len(t['y']), 256)
            self.assertEqual(len(np.unique(t['X'], axis=0)), 256)
            self.assertTrue(np.all(np.bincount(t['y'], minlength=C) >= 2))
            self.assertEqual(t['X'].dtype, np.float16)
            self.assertEqual(t['y'].dtype, np.uint8)

    def test_feasible_and_infeasible_resampling_flags(self):
        def skewed(rng, s, n):
            X, y = fake_dataset(rng, s, n)
            y[:] = 1
            y[:n//10] = 0
            return X, y
        for props, expected in [([.01,.99], True), ([.99,.01], True), ([.5,.5], False)]:
            with patch.object(gen, '_base_dataset', skewed), \
                    patch.object(gen, '_unpredictable', return_value=False), \
                    patch.object(gen, 'P_RESAMPLE', 1), \
                    patch.object(gen, '_target_proportions', return_value=np.array(props)):
                t = gen.generate_task(spec())
            self.assertEqual(t['resampled'], expected)
            self.assertTrue(t['resample_requested'])
            self.assertEqual(len(t['y']), 256)
            self.assertTrue(np.array_equal(np.unique(t['y']), np.arange(2)))

    def test_quotas_fuzz_and_invalid(self):
        rng = np.random.default_rng(42)
        for C in range(2,11):
            for n in (2*C, 2*C+1, 100, 257):
                for _ in range(25):
                    take = gen._quotas(rng.dirichlet(np.full(C,.02)), n)
                    self.assertEqual(take.sum(), n)
                    self.assertTrue(np.all(take >= 2))
        for p, n in [([1,1],3), ([0,0],10), ([np.nan,1],10), ([-1,2],10)]:
            with self.assertRaises(ValueError):
                gen._quotas(p,n)

    def test_bad_candidates_rejected_before_casts(self):
        s = spec(n=20,d=2,C=3)
        X = np.arange(120,dtype=np.float32).reshape(60,2)
        y = np.arange(60)%3
        variants = []
        for value in (np.nan, np.inf, -np.inf, 65520, 1001):
            xx = X.copy(); xx[0,0] = value
            variants.append((xx,y))
        # abs(min_signed_integer) overflows in its original dtype; range checks must
        # reject it before float16 conversion or the sklearn predictability filter.
        for dtype in (np.int16,np.int64):
            xx=X.astype(dtype);xx[0,0]=np.iinfo(dtype).min
            variants.append((xx,y))
        yy = y.copy(); yy[0]=-100; variants.append((X,yy))
        yy = y.astype(float); yy[0]=.5; variants.append((X,yy))
        yy = y.copy(); yy[0]=256; variants.append((X,yy))
        variants.extend([(np.ones_like(X),y), (X[:0],y[:0]), (X,y[:-1])])
        for xx,yy in variants:
            self.assertIsNone(gen._clean_candidate(xx,yy,s))
        clean = gen._clean_candidate(np.r_[X,X], np.r_[y,y],s)
        self.assertEqual(len(clean[1]),120)
        for constant in (np.ones(60), 1+np.arange(60)*1e-7):
            xx=X.copy();xx[:,0]=constant
            clean=gen._clean_candidate(xx,y,s)
            self.assertEqual(clean[0].shape,(60,1))
        xx=X.copy();xx[:,1]=xx[:,0]
        self.assertEqual(gen._clean_candidate(xx,y,s)[0].shape,(60,2))

    def test_constant_removal_and_duplicates_for_both_priors(self):
        def repeated(rng,s,n):
            col=np.arange(n)%4
            return np.column_stack([col,col,np.ones(n)]),np.arange(n)%2
        for prior in ("graph", "tree"):
            with patch.object(gen,'_base_dataset',repeated), \
                    patch.object(gen,'_unpredictable',return_value=False) as check:
                t=gen.generate_task(spec(n=64,d=3,prior=prior))
            self.assertEqual(t['X'].shape,(64,2))
            self.assertEqual(t['invalid'],0)
            self.assertLess(len(np.unique(t['X'],axis=0)),64)
            np.testing.assert_array_equal(t['X'][:,0],t['X'][:,1])
            self.assertEqual(check.call_args[0][0].shape[1],2)
            g=gen.assemble_group(np.random.default_rng(8),[t],.6)
            self.assertEqual(g['X'].shape,(1,64,3))
            self.assertEqual(g['d'].item(),2)
            self.assertTrue(torch.all(g['X'][:,:,2:]==0))
            self.assertTrue(np.isin(g['X'][0,g['n_train']:,0].numpy(),
                                    g['X'][0,:g['n_train'],0].numpy()).all())
            gen.validate_group(g)

    def test_constant_removal_after_selection_and_empty_redraw(self):
        class OrderedRng:
            def __init__(self,seed):self.rng=np.random.default_rng(seed)
            def __getattr__(self,key):return getattr(self.rng,key)
            def permutation(self,a):return np.arange(a) if np.isscalar(a) else np.asarray(a)
        def rare(rng,s,n):
            X=np.zeros((n,2));X[-1]=1
            if s['seed']==1:X[:,0]=np.arange(n)
            return X,np.arange(n)%2
        for seed in (0,1):
            sources=[rare,fake_dataset]
            def draw(rng,s,n):return sources.pop(0)(rng,s,n)
            with patch.object(gen,'_base_dataset',draw),patch.object(gen,'P_RESAMPLE',1), \
                    patch.object(gen,'_seed_task',return_value=OrderedRng(seed)), \
                    patch.object(gen,'_unpredictable',return_value=False) as check, \
                    patch.object(gen,'_target_proportions',return_value=np.array([.5,.5])), \
                    patch.object(gen,'_select_rows',return_value=np.arange(20)):
                t=gen.generate_task(spec(n=20,d=2,seed=seed))
            self.assertEqual(t['X'].shape[1],2 if seed==0 else 1)
            self.assertEqual(t['invalid'],1 if seed==0 else 0)
            self.assertEqual(check.call_count,2 if seed==0 else 1)

    def test_resampling_matches_rank_and_preserves_quota_multiset(self):
        def unbalanced(rng,s,n):
            X=rng.normal(size=(n,2))
            return X,np.repeat(np.arange(3),[10,110,80])
        for p in ([.8,.05,.15],[.15,.8,.05],[.05,.15,.8]):
            with patch.object(gen,'_base_dataset',unbalanced),patch.object(gen,'P_RESAMPLE',1), \
                    patch.object(gen,'_unpredictable',return_value=False), \
                    patch.object(gen,'_target_proportions',return_value=np.array(p)):
                t=gen.generate_task(spec(n=100,d=2,C=3))
            self.assertTrue(t['resampled'])
            np.testing.assert_array_equal(np.sort(np.bincount(t['y'])),np.sort(gen._quotas(p,100)))

    def test_class_shortages_are_rejected_before_filter(self):
        for C in (2,3,10):
            for missing in (True,False):
                def bad(rng,s,n):
                    X,y=fake_dataset(rng,s,n)
                    y[:]=0
                    for c in range(1,C):y[c]=c
                    if missing:y[y==C-1]=0
                    return X,y
                with patch.object(gen,'_base_dataset',bad),patch.object(gen,'MAX_TASK_DRAFTS',2), \
                        patch.object(gen,'_unpredictable') as check:
                    with self.assertRaisesRegex(RuntimeError,'structural retries exhausted'):
                        gen.generate_task(spec(C=C))
                self.assertFalse(check.called)

    def test_structural_exhaustion(self):
        def bad(rng,s,n):
            return np.ones((n,s['n_features'])), np.arange(n)%s['n_classes']
        with patch.object(gen, '_base_dataset', bad), patch.object(gen,'MAX_TASK_DRAFTS',3), \
                patch.object(gen, '_unpredictable') as check:
            with self.assertRaisesRegex(RuntimeError,'structural retries exhausted'):
                gen.generate_task(spec())
            self.assertFalse(check.called)

    def test_malformed_source_uses_structural_retries(self):
        for X,y in ((np.zeros((5,4)),np.arange(10)%2),
                    (np.zeros(5),np.arange(5)%2), (np.zeros((0,4)),np.zeros(0))):
            with patch.object(gen,'_base_dataset',return_value=(X,y)), \
                    patch.object(gen,'MAX_TASK_DRAFTS',2),patch.object(gen,'_unpredictable') as check:
                with self.assertRaisesRegex(RuntimeError,'structural retries exhausted'):
                    gen.generate_task(spec())
                self.assertFalse(check.called)

    def test_split_fuzz_and_infeasible(self):
        rng = np.random.default_rng(13)
        for C in range(2,11):
            for _ in range(30):
                counts = rng.integers(1,20,C)
                y = np.repeat(np.arange(C),counts)
                test_min = max(1,int((counts>=2).sum()))
                min_rows = C+test_min
                if min_rows > len(y):
                    continue
                n = int(rng.integers(min_rows,len(y)+1))
                ntr = int(rng.integers(C,n-test_min+1))
                sel = gen.stratified_split(rng,y,n,ntr)
                self.assertEqual(len(sel),n)
                self.assertEqual(len(np.unique(sel)),n)
                np.testing.assert_array_equal(np.unique(y[sel[:ntr]]),np.arange(C))
                self.assertTrue(np.isin(np.flatnonzero(counts>=2),y[sel[ntr:]]).all())
        for n,ntr in ((32,29),(5,1),(5,5),(100,10)):
            with self.assertRaises(ValueError):
                gen.stratified_split(rng,np.repeat(np.arange(10),4),n,ntr)

    def test_group_preserves_geometry_and_padding(self):
        tasks = [fake_task(seed=i,d=i%3+1,C=i%3+2) for i in range(8)]
        g = gen.assemble_group(np.random.default_rng(12),tasks,.9)
        gen.validate_group(g,preset=gen.PRESETS['s1b'],max_features=3,group_size=8)
        self.assertEqual(g['X'].shape,(8,256,3))
        self.assertEqual(g['y'].dtype,torch.uint8)
        broken = tasks.copy(); broken[0] = fake_task(n=128)
        with self.assertRaisesRegex(ValueError,'requested row count'):
            gen.assemble_group(np.random.default_rng(1),broken,.7)
        tiny = gen.assemble_group(np.random.default_rng(1),[fake_task(n=20,d=1,C=10)],.9)
        self.assertEqual(tiny['n_train'],10)
        self.assertEqual(len(torch.unique(tiny['y'][0,10:])),10)

    def test_validation_catches_corruption(self):
        good = gen.assemble_group(np.random.default_rng(1),[fake_task(d=1),fake_task(d=2)],.6)
        corruptions = [lambda g:g['X'].__setitem__((0,0,0),float('inf')),
            lambda g:g['X'].__setitem__((0,0,1),1),
            lambda g:g['y'].__setitem__((0,0),255),
            lambda g:g['n_classes'].__setitem__(0,16),
            lambda g:g.update(n_train=256),lambda g:g['d'].__setitem__(0,0),
            lambda g:g['X'].__setitem__((1,slice(None),0),1),
            lambda g:g['y'].__setitem__((1,slice(g['n_train'],None)),0)]
        for corrupt in corruptions:
            g=copy.deepcopy(good); corrupt(g)
            with self.assertRaises(ValueError): gen.validate_group(g)

    def test_geometry_and_seeds(self):
        rng=np.random.default_rng(4)
        for preset in gen.PRESETS.values():
            for maximum in (1,2,100):
                for geometry in ('ratio','cap'):
                    for i in range(80):
                        specs,frac=gen.group_specs(rng,preset,maximum,[1,2,i], 'graph',geometry,.55)
                        for s in specs:
                            self.assertTrue(preset['n_min']<=s['n_rows']<=preset['n_max'])
                            self.assertTrue(1<=s['n_features']<=maximum)
                            self.assertTrue(2<=s['n_classes']<=10)
                        self.assertTrue(preset['train_min']<=frac<=preset['train_max'])
        coords=[gen.coordinate_seed(2,p,s,g,t,domain) for p in gen.PRIORS for s in range(3)
                for g in (0,1,1000,1000003) for t in range(8) for domain in (1,2,3)]
        self.assertEqual(len({tuple(c) for c in coords}),len(coords))
        states=[tuple(np.random.SeedSequence(c).generate_state(4)) for c in coords]
        self.assertEqual(len(set(states)),len(coords))
        with self.assertRaises(ValueError):gen.coordinate_seed(2**32,'graph',0,0)
        with self.assertRaises(ValueError):gen.coordinate_seed(1.5,'graph',0,0)
        for bad in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):gen.group_specs(rng,gen.PRESETS['s1b'],100,0,'tree',p_binary=bad)

    def test_max_cells(self):
        # None draws exactly as without the argument; a cap keeps rows x features under it
        for preset in gen.PRESETS.values():
            for geometry in ('ratio','cap'):
                for i in range(50):
                    a=gen.group_specs(np.random.default_rng([7,i]),preset,100,[1,i],'graph',geometry,.55)
                    b=gen.group_specs(np.random.default_rng([7,i]),preset,100,[1,i],'graph',geometry,.55,None)
                    self.assertEqual(a,b)
                    specs,_=gen.group_specs(np.random.default_rng([7,i]),preset,100,[1,i],'graph',geometry,.55,800_000)
                    for s in specs:
                        self.assertTrue(2<=s['n_features']<=100)
                        self.assertTrue(s['n_rows']*s['n_features']<=max(800_000,2*s['n_rows']))
        widths=[gen.group_specs(np.random.default_rng([8,i]),gen.PRESETS['s4'],100,[1,i],'graph','cap',.55,800_000)[0][0]
                for i in range(400)]
        self.assertTrue(any(s['n_rows']<20_000 and s['n_features']>40 for s in widths))
        self.assertTrue(max(s['n_features'] for s in widths if s['n_rows']>50_000)<=16)

    def test_real_prior_determinism_and_small_width(self):
        for prior, version in (("graph", 3), ("graph", 4), ("tree", 4), ("rule", 1)):
            for d,C in ((1,2),(2,3)):
                s=spec(n=64,d=d,C=C,prior=prior,seed=8765+d)
                s["prior_version"] = version
                a,b=gen.generate_task(s),gen.generate_task(s)
                np.testing.assert_array_equal(a['X'],b['X'])
                np.testing.assert_array_equal(a['y'],b['y'])
                g=gen.assemble_group(np.random.default_rng(0),[a],.6)
                gen.validate_group(g)
        X,y=trees.generate_tree_task(np.random.default_rng(8),4,1,2)
        self.assertEqual(X.shape,(4,1)); self.assertTrue(np.isfinite(X).all())
        for n,d,C in ((3,1,2),(32,0,2),(32,1,11)):
            with self.assertRaises(ValueError):trees.generate_tree_task(np.random.default_rng(0),n,d,C)

    def test_tree_generator_fits_independent_rows(self):
        def continuous(rng,n,d):
            return np.tile(np.arange(n, dtype=float)[:,None], (1,d)), np.full(d,'normal')
        def fitted(rng,X_fit,T,X):
            self.assertFalse(np.isin(X[:,0],X_fit[:,0]).any())
            self.assertEqual(len(T),len(X_fit))
            return np.tile(np.linspace(-1,1,len(X))[:,None],(1,T.shape[1])), 'test'
        with patch.object(trees,'fit_trees',fitted), patch.object(trees,'sample_features',continuous):
            X,y=trees.generate_tree_task(np.random.default_rng(333),64,2,3)
            self.assertTrue(np.isfinite(X).all())
            self.assertTrue(np.all((y>=0)&(y<3)))


class ModernPriorTests(unittest.TestCase):
    def test_filter_reuses_effective_config_and_matches_brier_gate(self):
        from lightpfn.prior.v4 import filter_metrics
        rng = np.random.default_rng(19)
        cfg = gen._prior()
        for C in (2, 4):
            X = rng.normal(size=(512, 5)).astype(np.float16)
            y = np.arange(512) % C
            self.assertEqual(gen._unpredictable(X, y, cfg), filter_metrics(X, y, cfg)['reject'])
        with patch('lightpfn.prior.v4.filter_metrics', return_value={'reject': False}) as fit:
            gen._unpredictable(X, y, cfg, return_metrics=True)
            self.assertIs(fit.call_args.args[2], cfg)

    def test_quantization_ties_rare_levels_and_permuted_codes(self):
        from lightpfn.prior.v4 import quantize_column
        X = np.arange(4096, dtype=float)
        a = quantize_column(np.random.default_rng(1), X, 256, .8)
        b = quantize_column(np.random.default_rng(1), X, 256, .8)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 256)
        self.assertFalse(np.all(np.diff(a) >= 0))
        counts = np.unique(a, return_counts=True)[1]
        self.assertGreater(counts.max()/counts.min(), 10)
        tied = np.repeat(np.arange(12), 30)
        out = quantize_column(np.random.default_rng(3), tied, 256)
        self.assertLessEqual(len(np.unique(out)), 12)
        for x in np.unique(tied):
            self.assertEqual(len(np.unique(out[tied == x])), 1)
        for x, k in (([np.inf, 1], 2), ([1, 2], 257), ([], 2)):
            with self.assertRaises(ValueError):
                quantize_column(np.random.default_rng(0), x, k)

    def test_every_rule_family_and_multiclass_oracle(self):
        from lightpfn.prior.rules import _make_rule, evaluate_rule
        from lightpfn.prior.v4 import CONFIG
        rng = np.random.default_rng(45)
        for C in (2, 3, 8, 10):
            for family in CONFIG['rule_families']:
                for attempt in range(30):
                    X = rng.normal(size=(2048, 12)).astype(np.float16)
                    X[:, 0] = np.arange(len(X)) % 32
                    cat = np.arange(12) == 0
                    meta = _make_rule(rng, X, cat, C, family)
                    if meta is not None:
                        oracle = evaluate_rule(X, meta)
                        if len(np.unique(oracle)) == C:
                            break
                else:
                    self.fail(f'could not construct {family}/{C}')
                self.assertTrue(np.array_equal(np.unique(oracle), np.arange(C)))
                self.assertTrue(all(0 <= j < 12 for j in meta['drivers']))
                self.assertEqual(meta['family'], family if not (family == 'tree' and C > 8) else 'lookup')

    def test_pure_interactions_have_no_driver_marginal(self):
        from lightpfn.prior.rules import _make_rule, evaluate_rule
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(1)
        for C in (2, 3, 10):
            for family in ('xor', 'parity'):
                X = rng.normal(size=(8192, 100)).astype(np.float16)
                meta = _make_rule(rng, X, np.zeros(100, dtype=bool), C, family)
                y = evaluate_rule(X, meta)
                for j in meta['drivers']:
                    et = ExtraTreesClassifier(n_estimators=50, min_samples_leaf=10, n_jobs=1, random_state=0)
                    et.fit(X[:4096, j:j+1], y[:4096])
                    p = et.predict_proba(X[4096:, j:j+1])
                    auc = roc_auc_score(y[4096:], p[:, 1]) if C == 2 else roc_auc_score(y[4096:], p, multi_class='ovr')
                    self.assertLess(abs(auc-.5), .06)
                if C == 2:
                    self.assertTrue(all(.3 <= q <= .7 for q in meta['source_quantiles']))
                else:
                    self.assertEqual(meta['interaction_radix'], C)

    def test_rule_bypasses_et_and_persists_validated_metadata(self):
        s = spec(n=512, d=30, C=2, prior='rule', seed=4)
        with patch.object(gen, '_unpredictable', side_effect=AssertionError('ET forbidden')):
            t = gen.generate_task(s)
        group = gen.assemble_group(np.random.default_rng(5), [t], .7)
        self.assertEqual(group['filter_checks'].item(), 0)
        gen.validate_group(group)
        broken = copy.deepcopy(group)
        broken['rule_meta'][0]['drivers'] = [1000]
        with self.assertRaises(ValueError): gen.validate_group(broken)
        broken = copy.deepcopy(group)
        broken['rule_oracle'][0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'oracle'): gen.validate_group(broken)
        broken = copy.deepcopy(group)
        broken['cat_mask'][0] = True
        broken['d'][0] -= 1
        broken['X'][0, :, int(broken['d'][0]):] = 0
        with self.assertRaises(ValueError): gen.validate_group(broken)

    def test_unsupported_version_rejected(self):
        for prior, version in [('graph', 2), ('tree', 3), ('rule', 4)]:
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                gen.generate_task(dict(spec(prior=prior), prior_version=version))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=TMP)
        self.out=Path(self.tmp.name)
        self.args=argparse.Namespace(preset='s1b',prior='tree',geometry='ratio',p_binary=.55,
             out=str(self.out),n_tasks=32,groups_per_shard=2,max_features=5,n_jobs=1,seed=14,min_free_gb=0.)
        self.meta=gen.generator_meta(self.args)
        gen.atomic_write(self.out/'meta.json',lambda f:f.write(__import__('json').dumps(self.meta).encode()))

    def tearDown(self):self.tmp.cleanup()

    def groups(self,shard=0,n=2):
        class Serial:
            def map(self,fn,specs,chunksize):return [fn(s) for s in specs]
        with patch.object(gen,'_base_dataset',fake_dataset),patch.object(gen,'_unpredictable',return_value=False):
            return gen.make_shard(Serial(),self.args,shard,n)

    def save(self,i,groups):
        gen.atomic_write(self.out/f'shard_{i:06d}.pt',lambda f:torch.save(groups,f))

    def test_partial_extension_matches_one_shot(self):
        first=self.groups(n=1); full=self.groups(n=2)
        for k in first[0]:
            if torch.is_tensor(first[0][k]):torch.testing.assert_close(first[0][k],full[0][k],rtol=0,atol=0)
            else:self.assertEqual(first[0][k],full[0][k])
        self.save(0,first)
        self.assertEqual(gen.check_resume(self.out,self.meta),[1])
        (self.out/'shard_000001.pt.tmp').write_bytes(b'incomplete')
        self.save(0,full)
        self.assertEqual(gen.check_resume(self.out,self.meta),[2])
        self.save(1,self.groups(shard=1))
        self.assertEqual(gen.check_resume(self.out,self.meta),[2,2])

    def test_metadata_checked_without_shards(self):
        for key,value in [('seed',99),('geometry','cap'),('p_binary',.5),('prior_version',1),('groups_per_shard',3)]:
            bad=copy.deepcopy(self.meta); bad[key]=value
            with self.assertRaisesRegex(ValueError,'incompatible'):gen.check_resume(self.out,bad)
        bad=copy.deepcopy(self.meta);bad['versions']['torch']='other'
        with self.assertRaises(ValueError):gen.check_resume(self.out,bad)
        self.assertEqual(gen.check_resume(self.out,self.meta),[])

    def test_holes_corruption_missing_meta_and_shrink(self):
        full=self.groups()
        self.save(1,full)
        with self.assertRaisesRegex(ValueError,'contiguous'):gen.check_resume(self.out,self.meta)
        (self.out/'shard_000001.pt').unlink()
        self.save(0,full)
        smaller=copy.deepcopy(self.meta);smaller['n_tasks']=8
        with self.assertRaisesRegex(ValueError,'smaller'):gen.check_resume(self.out,smaller)
        (self.out/'meta.json').unlink()
        with self.assertRaisesRegex(ValueError,'without meta'):gen.check_resume(self.out,self.meta)
        (self.out/'meta.json').write_text(__import__('json').dumps(self.meta))
        (self.out/'shard_000000.pt').write_bytes(b'partial zip')
        with self.assertRaisesRegex(ValueError,'invalid committed shard'):gen.check_resume(self.out,self.meta)

    def test_only_final_shard_can_be_partial(self):
        self.save(0,self.groups(n=1));self.save(1,self.groups(shard=1))
        with self.assertRaisesRegex(ValueError,'only the last'):gen.check_resume(self.out,self.meta)

    def test_valid_shard_from_wrong_coordinates_is_rejected(self):
        self.save(0,self.groups(shard=1))
        with self.assertRaisesRegex(ValueError,'seeded specification'):gen.check_resume(self.out,self.meta)

    def test_resume_accepts_smaller_effective_width_but_requires_requested_tensor_width(self):
        groups=self.groups()
        for g in groups:
            self.assertGreater(g['X'].shape[2],1)
            g['X'][:,:,1:]=0;g['d'].fill_(1)
        self.save(0,groups)
        self.assertEqual(gen.check_resume(self.out,self.meta),[2])
        groups[0]['X']=groups[0]['X'][:,:,:1].contiguous()
        self.save(0,groups)
        with self.assertRaisesRegex(ValueError,'seeded specification'):gen.check_resume(self.out,self.meta)

    def test_resume_rejects_forged_diagnostics(self):
        for key in ('drafts','filter_checks','rejected','invalid','resample_requested'):
            groups=self.groups()
            if key=='resample_requested':
                groups[0]['resampled'][0]=True;groups[0][key][0]=False
            else:groups[0][key][0]+=1000
            self.save(0,groups)
            with self.assertRaises(ValueError):gen.check_resume(self.out,self.meta)
        for rejected,invalid,drafts in ((64,0,65),(0,128,129)):
            groups=self.groups();g=groups[0]
            for key,value in [('rejected',rejected),('invalid',invalid),('drafts',drafts),
                              ('filter_checks',rejected+1)]:g[key].fill_(value)
            self.save(0,groups)
            with self.assertRaisesRegex(ValueError,'diagnostics'):gen.check_resume(self.out,self.meta)

    def test_atomic_interruption_and_lock(self):
        path=self.out/'atomic'
        path.write_bytes(b'committed')
        def broken(f):f.write(b'partial');raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):gen.atomic_write(path,broken)
        self.assertEqual(path.read_bytes(),b'committed')
        with gen.pool_lock(self.out):
            with self.assertRaisesRegex(ValueError,'holds the pool lock'):
                with gen.pool_lock(self.out):pass
        with gen.pool_lock(self.out):pass

    def test_real_cli_resume_cross_process_hash_seed_and_worker_count(self):
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        for prior, version in (("graph", 3), ("graph", 4), ("tree", 4), ("rule", 1)):
            a,b=self.out/(prior+str(version)+'direct'),self.out/(prior+str(version)+'resume')
            def run(out,n,jobs,hash_seed):
                cmd=[sys.executable,'-m','lightpfn.prior.generate','--preset','s1b','--prior',prior,'--prior-version',str(version),
                     '--geometry','ratio','--p-binary','.55','--out',str(out),'--n-tasks',str(n),
                     '--groups-per-shard','2','--max-features','4','--n-jobs',str(jobs),
                     '--seed','123','--min-free-gb','0']
                result=subprocess.run(cmd,cwd=root,env=dict(env,PYTHONHASHSEED=str(hash_seed)),
                                      capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run(a,24,1,1)
            run(b,8,2,9876)
            (b/'shard_000001.pt.tmp').write_bytes(b'interrupted')
            run(b,24,2,9876)
            for f in a.glob('shard_*.pt'):
                self.assertEqual(f.read_bytes(),(b/f.name).read_bytes())
            run(b,24,1,7)  # completed pool is a validated no-op
            # Captured before this change, same dependency versions and seed/config.
            golden = {("graph", 3): ["2625d88bac2891ebc442f20e27dff4da008c8e4582848105fcf6e88178c3c3fb", "5fbfff49e9f7d43aee2c2e80469edb2bf6f6e41a37da532c406b4e9a6944b490"],
                      ("tree", 4): ["7a4f3d6250bb1b7ef3d29a52c58b94e26e03014feb6394406558d0e8b6ec569e", "c7f8131a3d184bfd44207eed6b96b7078f9d88eb6dd69def07440194ce26cacb"]}
            if (prior, version) in golden and torch.__version__ == "2.13.0+rocm7.2" and np.__version__ == "2.5.2":
                import hashlib
                self.assertEqual([hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.glob("shard_*.pt"))], golden[prior, version])


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)
