"""Chunked stream (lightpfn.prior.stream) and epoch sampling (PoolStream). CPU only, Linux (fcntl).

The generator and trainer are replaced by fakes that write what the real commands would write,
so the waiting, retry, cleanup and resume logic runs in milliseconds.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightpfn.prior import stream as st
from lightpfn.train import PoolStream

PRIORS = [dict(label="v3", prior="graph", version=3, p_binary=.55, weight=.9, seed_base=9_000_000),
          dict(label="rule", prior="rule", version=1, p_binary=.78, weight=.1, seed_base=9_500_000)]


def init(tmp_path, chunks=5, consumers=("a",), chunk_steps=100):
    spec = st.make_spec(chunk_steps, chunks, list(consumers), [dict(p) for p in PRIORS])
    s = tmp_path / "stream"
    s.mkdir()
    st.atomic_json(s / "spec.json", spec)
    return s, spec


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class FakeGenerator:
    """Writes meta.json and the expected shard files (or fewer, to simulate a disk stop)."""

    def __init__(self, stream, short=0):
        self.stream, self.short, self.calls, self.max_chunks_on_disk = stream, short, [], 0

    def __call__(self, cmd, **kw):
        out = Path(arg(cmd, "--out"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text("{}")
        n = math.ceil(int(arg(cmd, "--n-tasks")) / st.SHARD_TASKS) - self.short
        for i in range(n):
            (out / f"shard_{i:06d}.pt").write_bytes(b"x")
        self.calls.append((out.parent.name, out.name, int(arg(cmd, "--seed"))))
        on_disk = [d for d in self.stream.iterdir() if st.CHUNK_RE.fullmatch(d.name)]
        self.max_chunks_on_disk = max(self.max_chunks_on_disk, len(on_disk))
        return SimpleNamespace(returncode=0)


def test_spec_sizes_seeds_and_validation():
    spec = st.make_spec(8000, 20, ["F1", "F2"], [dict(p) for p in PRIORS])
    v3, rule = spec["priors"]
    assert v3["tasks"] % st.SHARD_TASKS == 0 and rule["tasks"] % st.SHARD_TASKS == 0
    assert v3["tasks"] >= .9 * 8000 * 32 * 1.05 > v3["tasks"] - st.SHARD_TASKS
    assert abs(v3["weight"] + rule["weight"] - 1) < 1e-12 and spec["total_steps"] == 160_000
    twice = st.make_spec(8000, 20, ["F1"], [dict(p) for p in PRIORS], passes=2)["priors"][0]
    assert twice["tasks"] == math.ceil(.9 * 8000 * 32 * 1.05 / 2 / st.SHARD_TASKS) * st.SHARD_TASKS
    bad = [dict(PRIORS[0], seed_base=0)], [dict(PRIORS[0], seed_base=0xD1DE0001 - 3)], \
          [PRIORS[0], dict(PRIORS[1], seed_base=9_000_010)], [PRIORS[0], dict(PRIORS[1], label="v3")]
    for priors in bad:
        with pytest.raises(ValueError):
            st.make_spec(8000, 20, ["F1"], [dict(p) for p in priors])
    with pytest.raises(ValueError):
        st.make_spec(8000, 20, ["F1", "F1"], [dict(PRIORS[0])])
    assert st.parse_prior("v4=graph:4:0.78:0.45:4000000") == dict(
        label="v4", prior="graph", version=4, p_binary=.78, weight=.45, seed_base=4_000_000)
    with pytest.raises(ValueError):
        st.parse_prior("v4=graph:4:0.78:0.45")
    typed = st.parse_prior("v3=graph:3:0.55:0.9:3000000:typed")
    assert typed == dict(label="v3", prior="graph", version=3, p_binary=.55, weight=.9, seed_base=3_000_000, typed=True)
    with pytest.raises(ValueError):  # level IDs exist for the graph prior only
        st.parse_prior("rule=rule:1:0.78:0.1:5000000:typed")


def test_typed_prior_reaches_the_generator_and_the_carry_over_check(tmp_path):
    spec = st.make_spec(100, 2, ["a"], [st.parse_prior("v3=graph:3:0.55:0.9:9000000:typed"), dict(PRIORS[1])])
    v3, rule = spec["priors"]
    assert v3["typed"] and "typed" not in rule
    assert st.gen_command(spec, v3, tmp_path / "v3", 9_000_000, 4, 20)[-1] == "--typed"
    assert "--typed" not in st.gen_command(spec, rule, tmp_path / "rule", 9_500_000, 4, 20)
    out = tmp_path / "v3"
    out.mkdir()
    for i in range(math.ceil(v3["tasks"] / st.SHARD_TASKS)):
        (out / f"shard_{i:06d}.pt").write_bytes(b"x")
    meta = dict(seed=9_000_000, prior="graph", prior_version=3, p_binary=.55, preset="s1b", geometry="ratio",
                max_features=100)
    for typed, ok in ((None, False), (False, False), (True, True)):  # an untyped pool is not a typed chunk
        (out / "meta.json").write_text(json.dumps(meta if typed is None else dict(meta, typed=typed)))
        assert st.carried_over(out, v3, 9_000_000, spec) is ok
    untyped = dict(v3)
    del untyped["typed"]
    (out / "meta.json").write_text(json.dumps(meta))  # pools from before --typed have no key
    assert st.carried_over(out, untyped, 9_000_000, spec)


def test_until_stops_generator_and_consumer_with_the_schedule_unchanged(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=4, consumers=("R",))
    fake = FakeGenerator(s)
    monkeypatch.setattr(st, "ROOT", tmp_path)
    # The fake generator writes tiny files; real free space (e.g. a 16 GiB /tmp
    # tmpfs) must not trigger the production 30 GB disk-space wait loop.
    monkeypatch.setattr(st, "free_gb", lambda p: 1e6)
    (tmp_path / "runs" / "train" / "R").mkdir(parents=True)
    # the consumer has not started: the generator stays `ahead` chunks in front of chunk 0 and stops at `until`
    assert st.generate(s, ahead=4, run=fake, sleep=lambda _: None, until=2) == 0
    assert sorted({c[0] for c in fake.calls}) == ["c000000", "c000001"] and not (s / "GEN_DONE").exists()
    cmds = []

    def fake_train(cmd, **kw):
        cmds.append(cmd)
        (tmp_path / "runs" / "train" / "R" / f"ema_step{int(arg(cmd, '--stop-at')):06d}.pt").write_bytes(b"x")
        return SimpleNamespace(returncode=0)

    evals = []
    assert st.consume(s, "R", 0, [], eval_script="e.sh", run=fake_train, popen=lambda c, **kw: evals.append(c[-2]),
                      sleep=lambda _: None, until=2) == 0
    assert [(arg(c, "--steps"), arg(c, "--stop-at")) for c in cmds] == [("400", "100"), ("400", "200")]
    assert evals == ["R_s100", "R_s200"]
    state = st.consumer_state(s, "R")
    assert state["done"] == 2 and state["state"] == "paused" and state["chunks"] == 4
    # a paused consumer still holds its next chunk; the consumed ones are gone
    assert st.floor_chunk(s, spec) == 2
    assert not any((s / f"c{k:06d}").exists() for k in (0, 1))
    with pytest.raises(ValueError):
        st.consume(s, "R", 0, [], run=fake_train, sleep=lambda _: None, until=5)
    # rerun without until: the generator and the consumer continue to the end of the same schedule
    assert st.generate(s, ahead=4, run=fake, sleep=lambda _: None) == 0 and (s / "GEN_DONE").exists()
    assert st.consume(s, "R", 0, [], run=fake_train, sleep=lambda _: None) == 0
    assert [arg(c, "--stop-at") for c in cmds[2:]] == ["300", "400"] and st.consumer_state(s, "R")["state"] == "done"


def test_generate_stays_ahead_cleans_and_finishes(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=6)
    monkeypatch.setattr(st, "free_gb", lambda p: 1e6)
    gen = FakeGenerator(s)

    def consumer_advances(_):  # the consumer finishes one chunk per wait
        state = st.consumer_state(s, "a")
        st.write_consumer(s, "a", dict(state, done=state["done"] + 1))

    assert st.generate(s, ahead=2, poll=1, run=gen, sleep=consumer_advances) == 0
    chunks = [c for c, label, _ in gen.calls if label == "v3"]
    assert chunks == [f"c{k:06d}" for k in range(6)]
    assert gen.max_chunks_on_disk <= 2 + 1  # `ahead` ready chunks plus the one being written
    assert {(c, label): seed for c, label, seed in gen.calls}[("c000004", "rule")] == 9_500_004
    assert (s / "GEN_DONE").exists()
    # chunks finished by the consumer are gone, the others are READY
    done = st.consumer_state(s, "a")["done"]
    left = sorted(d.name for d in s.iterdir() if st.CHUNK_RE.fullmatch(d.name))
    assert left == [f"c{k:06d}" for k in range(done, 6)]
    assert all((s / c / "READY").exists() for c in left)


def test_generate_waits_for_disk(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=1)
    free = iter([1.0, 1.0, 1e6, 1e6])
    monkeypatch.setattr(st, "free_gb", lambda p: next(free))
    sleeps = []
    assert st.generate(s, poll=1, run=FakeGenerator(s), sleep=sleeps.append) == 0
    assert len(sleeps) == 2 and (s / "c000000" / "READY").exists()


def test_generate_retries_incomplete_pool_then_fails(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=2)
    monkeypatch.setattr(st, "free_gb", lambda p: 1e6)
    gen = FakeGenerator(s, short=1)
    assert st.generate(s, poll=1, retries=2, run=gen, sleep=lambda _: None) == 1
    assert len(gen.calls) == 3 and (s / "GEN_FAILED").exists()
    assert not (s / "c000000" / "READY").exists()
    # a rerun is a retry and clears the failure marker
    assert st.generate(s, poll=1, run=FakeGenerator(s), sleep=lambda _: None) == 0
    assert not (s / "GEN_FAILED").exists() and (s / "GEN_DONE").exists()


def test_pool_with_extra_shards_is_complete(tmp_path):
    out = tmp_path / "p"
    out.mkdir()
    (out / "meta.json").write_text("{}")
    for i in range(5):
        (out / f"shard_{i:06d}.pt").write_bytes(b"x")
    assert st.pool_complete(out, 4 * st.SHARD_TASKS) and st.pool_complete(out, 5 * st.SHARD_TASKS)
    assert not st.pool_complete(out, 6 * st.SHARD_TASKS)


def test_generate_keeps_a_carried_over_pool(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=1)
    monkeypatch.setattr(st, "free_gb", lambda p: 1e6)
    v3, rule = spec["priors"]
    meta = dict(seed=v3["seed_base"], prior="graph", prior_version=3, p_binary=.55, preset="s1b", geometry="ratio",
                max_features=100, n_tasks=4 * v3["tasks"])
    for label, m in (("v3", meta), ("rule", dict(meta, seed=rule["seed_base"] + 1, prior="rule", prior_version=1,
                                                       p_binary=.78))):
        out = s / "c000000" / label
        out.mkdir(parents=True)
        (out / "meta.json").write_text(json.dumps(m))
        for i in range(v3["tasks"] // st.SHARD_TASKS + 3):  # more shards than the chunk needs
            (out / f"shard_{i:06d}.pt").write_bytes(b"x")
    gen = FakeGenerator(s)
    assert st.generate(s, poll=1, run=gen, sleep=lambda _: None) == 0
    # v3 matches this chunk (seed, prior, settings): kept; rule has another seed: regenerated
    assert gen.calls == [("c000000", "rule", rule["seed_base"])]
    assert (s / "c000000" / "READY").exists()


def test_generate_rerun_skips_ready_and_consumed_chunks(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=4)
    monkeypatch.setattr(st, "free_gb", lambda p: 1e6)
    st.write_consumer(s, "a", dict(done=1, state="running", waited_s=0))
    (s / "c000001").mkdir()
    st.atomic_json(s / "c000001" / "READY", {})
    gen = FakeGenerator(s)
    assert st.generate(s, ahead=10, poll=1, run=gen, sleep=lambda _: None) == 0
    assert sorted({c for c, _, _ in gen.calls}) == ["c000002", "c000003"]


def test_generate_stops_when_every_consumer_failed(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=3, consumers=("a", "b"))
    monkeypatch.setattr(st, "free_gb", lambda p: 1e6)
    for r in ("a", "b"):
        st.write_consumer(s, r, dict(done=0, state="failed", waited_s=0))
    gen = FakeGenerator(s)
    assert st.generate(s, poll=1, run=gen, sleep=lambda _: None) == 0
    assert gen.calls == [] and not (s / "GEN_DONE").exists()


def ready_all(s, spec):
    for k in range(spec["chunks"]):
        (s / f"c{k:06d}").mkdir(exist_ok=True)
        st.atomic_json(s / f"c{k:06d}" / "READY", {})


def test_consume_segments_marks_cleans_and_evaluates(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=3, consumers=("R",))
    ready_all(s, spec)
    monkeypatch.setattr(st, "ROOT", tmp_path)
    (tmp_path / "runs" / "train" / "R").mkdir(parents=True)
    cmds, evals = [], []

    def fake_train(cmd, **kw):
        cmds.append(cmd)
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        stop = int(arg(cmd, "--stop-at"))
        if stop != 200:  # eval every 100 steps except the middle one
            (tmp_path / "runs" / "train" / "R" / f"ema_step{stop:06d}.pt").write_bytes(b"x")
        return SimpleNamespace(returncode=0)

    rc = st.consume(s, "R", 1, ["--warmup", "10", "--config", '{"row_mode":"summary"}'], eval_script="runs/e.sh",
                    run=fake_train, popen=lambda c, **kw: evals.append(c[-2:]), sleep=lambda _: None)
    assert rc == 0
    assert [int(arg(c, "--stop-at")) for c in cmds] == [100, 200, 300]
    assert all(arg(c, "--steps") == "300" and arg(c, "--sampling") == "epoch" for c in cmds)
    pools = cmds[1][cmds[1].index("--pools") + 1:cmds[1].index("--steps")]
    assert pools == [f"{s / 'c000001' / 'v3'}:0.9", f"{s / 'c000001' / 'rule'}:0.1"]
    assert cmds[0][-4:] == ["--warmup", "10", "--config", '{"row_mode":"summary"}']
    assert [e[0] for e in evals] == ["R_s100", "R"]
    state = st.consumer_state(s, "R")
    assert state["done"] == 3 and state["state"] == "done"
    assert not any(st.CHUNK_RE.fullmatch(d.name) for d in s.iterdir())
    # rerun: nothing left to do
    assert st.consume(s, "R", 1, [], run=fake_train, sleep=lambda _: None) == 0 and len(cmds) == 3


def test_consumers_with_different_lengths_share_chunks(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=4, consumers=("S", "L"))
    ready_all(s, spec)
    monkeypatch.setattr(st, "ROOT", tmp_path)
    (tmp_path / "runs" / "train" / "S").mkdir(parents=True)
    (tmp_path / "runs" / "train" / "S" / "ema_step000200.pt").write_bytes(b"x")
    cmds, evals = [], []
    ok = lambda c, **kw: cmds.append(c) or SimpleNamespace(returncode=0)  # noqa: E731
    assert st.consume(s, "S", 0, [], eval_script="e.sh", chunks=2, run=ok,
                      popen=lambda c, **kw: evals.append(c[-2]), sleep=lambda _: None) == 0
    assert [(arg(c, "--steps"), arg(c, "--stop-at")) for c in cmds] == [("200", "100"), ("200", "200")]
    assert evals == ["S"]  # the short run's last checkpoint carries its plain name
    state = st.consumer_state(s, "S")
    assert state["state"] == "done" and state["done"] == 2 and state["chunks"] == 2
    # the finished short run no longer holds chunks: the long one decides what is deleted
    st.write_consumer(s, "L", dict(done=3, state="running", waited_s=0))
    assert st.cleanup(s, spec) == 3
    assert sorted(d.name for d in s.iterdir() if st.CHUNK_RE.fullmatch(d.name)) == ["c000003"]
    with pytest.raises(ValueError):  # a restart cannot change the schedule length
        st.consume(s, "S", 0, [], chunks=3, run=ok, sleep=lambda _: None)
    with pytest.raises(ValueError):
        st.consume(s, "L", 0, [], chunks=5, run=ok, sleep=lambda _: None)


def test_consume_retries_then_fails_and_releases_its_chunks(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=2, consumers=("R", "other"))
    ready_all(s, spec)
    monkeypatch.setattr(st, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    st.write_consumer(s, "other", dict(done=2, state="done", waited_s=0))
    calls = []
    rc = st.consume(s, "R", 0, [], retries=2, run=lambda c, **kw: calls.append(c) or SimpleNamespace(returncode=1),
                    sleep=lambda _: None)
    assert rc == 1 and len(calls) == 3
    assert st.consumer_state(s, "R")["state"] == "failed"
    assert not any(st.CHUNK_RE.fullmatch(d.name) for d in s.iterdir())  # nobody holds them any more
    assert st.consume(s, "R", 0, [], run=lambda c, **kw: 1 / 0, sleep=lambda _: None) == 1  # stays failed


def test_consume_waits_for_ready_and_stops_on_generator_failure(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=2, consumers=("R",))
    monkeypatch.setattr(st, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    waits = []

    def sleep(_):
        waits.append(1)
        if len(waits) == 3:
            (s / "c000000").mkdir()
            st.atomic_json(s / "c000000" / "READY", {})
        if len(waits) == 5:
            (s / "GEN_FAILED").write_text("x")

    trained = []
    rc = st.consume(s, "R", 0, [], run=lambda c, **kw: trained.append(c) or SimpleNamespace(returncode=0), sleep=sleep)
    assert rc == 1 and len(trained) == 1
    state = st.consumer_state(s, "R")
    assert state["done"] == 1 and state["state"] == "stopped" and state["waited_s"] == 3 * 60
    with pytest.raises(ValueError):
        st.consume(s, "R", 0, ["--steps", "5"], run=None, sleep=None)
    with pytest.raises(ValueError):
        st.consume(s, "nobody", 0, [], run=None, sleep=None)


def marked_group(marker, B=2, n=16, m=3):
    X = np.full((B, n, m), marker, dtype=np.float16)
    y = np.tile(np.arange(n) % 2, (B, 1)).astype(np.uint8)
    return dict(X=torch.from_numpy(X), y=torch.from_numpy(y), d=torch.full((B,), m, dtype=torch.int16),
                n_classes=torch.full((B,), 2, dtype=torch.int16), n_train=8,
                resampled=torch.zeros(B, dtype=torch.bool))


def write_pool(path, n_shards, groups_per_shard, offset=0):
    path.mkdir(parents=True)
    for s in range(n_shards):
        torch.save([marked_group(offset + 10 * s + g) for g in range(groups_per_shard)], path / f"shard_{s:06d}.pt")
    return {offset + 10 * s + g for s in range(n_shards) for g in range(groups_per_shard)}


def marker(g):
    return int(round(float(g["X"][0, 0, 0])))


def test_epoch_sampling_reads_every_group_once_per_epoch(tmp_path):
    expected = write_pool(tmp_path / "p", 6, 3)
    it = iter(PoolStream([(tmp_path / "p", 1)], p_missing=0, seed=5, sampling="epoch"))
    first = [marker(next(it)) for _ in range(18)]
    second = [marker(next(it)) for _ in range(18)]
    assert sorted(first) == sorted(expected) and sorted(second) == sorted(expected)
    assert first != second
    again = iter(PoolStream([(tmp_path / "p", 1)], p_missing=0, seed=5, sampling="epoch"))
    assert [marker(next(again)) for _ in range(18)] == first
    with pytest.raises(ValueError):
        PoolStream([(tmp_path / "p", 1)], sampling="sometimes")


def test_epoch_sampling_mixes_pools_without_repeats(tmp_path):
    a = write_pool(tmp_path / "a", 40, 2)
    b = write_pool(tmp_path / "b", 40, 2, offset=500)  # markers stay below the |x| <= 1000 shard check
    it = iter(PoolStream([(tmp_path / "a", .5), (tmp_path / "b", .5)], p_missing=0, seed=3, sampling="epoch"))
    seen = [marker(next(it)) for _ in range(100)]
    assert len(set(seen)) == len(seen)  # 100 draws from 160 groups: no group twice
    assert 25 < sum(x >= 500 for x in seen) < 75 and set(seen) <= a | b


def test_epoch_sampling_workers_take_disjoint_slices(tmp_path):
    expected = write_pool(tmp_path / "p", 8, 2)

    def to_numpy(g):
        return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in g.items()}

    loader = torch.utils.data.DataLoader(PoolStream([(tmp_path / "p", 1)], p_missing=0, seed=11, sampling="epoch"),
                                         batch_size=None, num_workers=2, collate_fn=to_numpy)
    it = iter(loader)
    try:
        seen = [int(round(float(next(it)["X"][0, 0, 0]))) for _ in range(16)]
    finally:
        it._shutdown_workers()
    assert sorted(seen) == sorted(expected)


def test_epoch_sampling_ranks_take_disjoint_slices(tmp_path):
    expected = write_pool(tmp_path / "p", 8, 2)
    seen = []
    for rank in (0, 1):  # one process per rank, no DataLoader workers: slots rank::2
        it = iter(PoolStream([(tmp_path / "p", 1)], p_missing=0, seed=4, sampling="epoch", rank=rank, world=2))
        seen += [marker(next(it)) for _ in range(8)]
    assert sorted(seen) == sorted(expected)
    with pytest.raises(ValueError):
        PoolStream([(tmp_path / "p", 1)], rank=2, world=2)


def _ddp_worker(rank, init_file):
    import torch.distributed as dist
    from lightpfn.train import allreduce_grads, replicas_differ, sync_replicas
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    a, b, c = (torch.nn.Parameter(torch.ones(n)) for n in (3, 2, 1))
    a.grad = torch.full((3,), float(rank + 1))
    if rank == 1:  # b is used on one rank only, c on none
        b.grad = torch.full((2,), 5.0)
    allreduce_grads([a, b, c])
    assert torch.equal(a.grad, torch.full((3,), 3.0))
    assert torch.equal(b.grad, torch.full((2,), 5.0)) and c.grad is None
    torch.manual_seed(rank)
    m = torch.nn.Linear(4, 3)
    assert replicas_differ(m)
    sync_replicas([m])
    assert not replicas_differ(m)
    torch.manual_seed(0)
    assert torch.equal(m.weight, torch.nn.Linear(4, 3).weight)  # rank 0's values everywhere
    dist.destroy_process_group()


def test_data_parallel_helpers_with_gloo(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / "init"),), nprocs=2)


def test_stream_launches_torchrun_for_data_parallel(tmp_path, monkeypatch):
    s, spec = init(tmp_path, chunks=1, consumers=("R",))
    cmd = st.train_command(s, spec, "R", 0, ["--seed", "0"], nproc=2)
    assert cmd[1:6] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m"]
    assert cmd[6] == "lightpfn.train" and arg(cmd, "--sampling") == "epoch"
    with pytest.raises(ValueError):
        st.consume(s, "R", "0", [], nproc=2, run=None, sleep=None)
    assert st.make_spec(100, 2, ["R"], [dict(PRIORS[0])], tasks_per_step=64)["priors"][0]["tasks"] == \
        math.ceil(100 * 64 * 1.05 / st.SHARD_TASKS) * st.SHARD_TASKS


def test_replace_sampling_is_unchanged(tmp_path):
    """The default stream must match the pre-epoch implementation draw for draw."""
    write_pool(tmp_path / "p", 5, 2)
    it = iter(PoolStream([(tmp_path / "p", 1)], p_missing=0, seed=8))
    got = [marker(next(it)) for _ in range(12)]
    rng = np.random.default_rng([8, 0])
    ref, pending = [], []
    while len(ref) < 12:
        rng.choice(1, p=[1.0])
        if not pending:
            s = int(rng.integers(5))
            pending = [10 * s + g for g in rng.permutation(2)]
        ref.append(pending.pop())
        for _ in range(2):  # augment: one column permutation and one missingness draw per task
            rng.permutation(3)
            rng.random()
    assert got == ref


def test_stream_cli_train_passes_arguments(tmp_path, monkeypatch):
    """The exact command line of stream_launch.sh / stream_pilot.sh reaches consume() intact."""
    got = {}
    monkeypatch.setattr(st, "consume", lambda *a, **kw: got.update(args=a, kw=kw) and 0)
    argv = ["train", str(tmp_path), "--run", "P1", "--gpu", "0,1", "--nproc", "2", "--eval-script", "e.sh",
            "--chunks", "3", "--until", "2", "--", "--warmup", "20", "--config", '{"row_mode":"summary"}', "--seed", "0"]
    st.main(argv)
    stream, run, gpu, train_args, eval_script, chunks, nproc = got["args"]
    assert (stream, run, gpu, eval_script, chunks, nproc) == (tmp_path.resolve(), "P1", "0,1", Path("e.sh"), 3, 2)
    assert got["kw"] == dict(until=2)
    assert train_args == ["--warmup", "20", "--config", '{"row_mode":"summary"}', "--seed", "0"]
    with pytest.raises(SystemExit):
        st.main(["status", str(tmp_path), "--", "--seed", "0"])


def test_stream_cli_init_is_idempotent(tmp_path):
    s = tmp_path / "s"
    args = ["init", str(s), "--chunk-steps", "100", "--chunks", "3", "--consumers", "A",
            "--prior", "v3=graph:3:0.55:1:9000000"]
    assert st.main(args) == 0 and st.main(args) == 0
    assert json.loads((s / "spec.json").read_text())["chunks"] == 3
    with pytest.raises(SystemExit):
        st.main(args[:5] + ["4"] + args[6:])


@pytest.mark.parametrize("typed", [False, True])
def test_classification_spec_and_generator_command_are_byte_unchanged(tmp_path, typed):
    priors = [dict(p) for p in PRIORS]
    if typed:
        priors[0]["typed"] = True
    expected = dict(chunk_steps=100, chunks=2, total_steps=200, tasks_per_step=32, passes=1, consumers=["a"],
                    priors=[dict(priors[0], tasks=3072), dict(priors[1], tasks=512)],
                    margin=.05, preset="s1b", geometry="ratio", max_features=100)
    old_bytes = json.dumps(expected, indent=2)
    spec = st.make_spec(100, 2, ["a"], priors)
    assert json.dumps(spec, indent=2) == old_bytes
    assert json.dumps(st.make_spec(100, 2, ["a"], priors, task="classification"), indent=2) == old_bytes
    (tmp_path / "spec.json").write_text(old_bytes)
    assert st.load_spec(tmp_path) == expected
    older = dict(spec)
    older.pop("total_steps")
    assert arg(st.train_command(tmp_path, older, "a", 1, []), "--steps") == "200"
    item = spec["priors"][0]
    assert st.gen_command(spec, item, tmp_path / "v3", 9000000, 4, 20) == [
        st.sys.executable, "-m", "lightpfn.prior.generate", "--preset", "s1b", "--prior", "graph",
        "--prior-version", "3", "--geometry", "ratio", "--p-binary", "0.55", "--max-features", "100",
        "--seed", "9000000", "--n-tasks", "3072", "--n-jobs", "4", "--min-free-gb", "20", "--lookahead", "8",
        "--out", str(tmp_path / "v3"), *(["--typed"] if typed else [])]


def test_regression_spec_cli_roundtrip_and_generator_command(tmp_path):
    path = tmp_path / "stream"
    factors = "filter,noise,tails,discrete"
    argv = ["init", str(path), "--chunk-steps", "1024", "--chunks", "1", "--consumers", "R",
            "--margin", "0", "--tasks-per-step", "64", "--passes", "2", "--task", "regression",
            "--reg-factors", factors, "--prior", "r2=graph:3:0.4:1:3700000"]
    assert st.main(argv) == st.main(argv) == 0
    spec = st.load_spec(path)
    assert spec["task"] == "regression" and spec["reg_factors"] == factors.split(",")
    assert spec["priors"][0]["tasks"] == 32768
    assert st.gen_command(spec, spec["priors"][0], tmp_path / "r2", 3700000, 4, 10) == [
        st.sys.executable, "-u", "-m", "lightpfn.prior.generate", "--preset", "s1b",
        "--prior", "graph", "--prior-version", "3", "--task", "regression",
        "--geometry", "ratio", "--max-features", "100", "--seed", "3700000",
        "--n-tasks", "32768", "--n-jobs", "4", "--min-free-gb", "10",
        "--lookahead", "8", "--out", str(tmp_path / "r2"), "--reg-factors", factors,
    ]


def test_regression_generator_matches_local_r2_ab_command(tmp_path, monkeypatch):
    # The analysis script lives under gitignored runs/ and is unavailable in CI.
    r2_ab = pytest.importorskip("runs.analysis_reg_prior_20261105.r2_ab",
                                reason="local analysis script is not checked into Git")
    factors = "filter,noise,tails,discrete"
    spec = st.make_spec(1024, 1, ["R"], [st.parse_prior("r2=graph:3:0.4:1:3700000")],
                        margin=0, tasks_per_step=64, passes=2, task="regression", reg_factors=factors)
    assert spec["priors"][0]["tasks"] == r2_ab.TASKS
    got = []
    monkeypatch.setattr(r2_ab, "pool", lambda arm: tmp_path / "r2")
    monkeypatch.setattr(r2_ab, "run", lambda cmd, name: got.append(list(map(str, cmd))))
    monkeypatch.setattr(r2_ab, "log", lambda msg: None)
    r2_ab.generate("r2v3")
    assert st.gen_command(spec, spec["priors"][0], tmp_path / "r2", r2_ab.SEED, r2_ab.JOBS, 10) == got[0]

def test_regression_carry_over_does_not_accept_classification_p0_or_other_factors(tmp_path):
    p = st.parse_prior("r2=graph:3:0.4:1:9000000")
    spec = st.make_spec(1, 1, ["a"], [p], task="regression", reg_factors="filter,noise,tails,discrete")
    item = spec["priors"][0]
    (tmp_path / "shard_000000.pt").write_bytes(b"x")
    meta = dict(seed=p["seed_base"], prior="graph", prior_version=3, p_binary=.4, preset="s1b", geometry="ratio",
                max_features=100, typed=True, task="regression", reg_factors=["discrete", "filter", "noise", "tails"])
    for overrides, ok in (({}, True), ({"reg_factors": []}, False), ({"reg_factors": ["filter"]}, False),
                          ({"task": "classification"}, False), ({"typed": False}, False)):
        (tmp_path / "meta.json").write_text(json.dumps(dict(meta, **overrides)))
        assert st.carried_over(tmp_path, item, p["seed_base"], spec) is ok
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    cls_spec = st.make_spec(1, 1, ["a"], [p])
    assert not st.carried_over(tmp_path, item, p["seed_base"], cls_spec)
    for kwargs in (dict(task="classification", reg_factors="filter"), dict(task="regression", reg_factors="x"),
                   dict(task="regression", reg_factors="filter,filter")):
        with pytest.raises(ValueError):
            st.make_spec(1, 1, ["a"], [p], **kwargs)
    with pytest.raises(ValueError):
        st.make_spec(1, 1, ["a"], [dict(p, prior="rule")], task="regression")


def test_partial_last_chunk_keeps_one_cosine_and_rounds_only_data(tmp_path, monkeypatch):
    s, _ = init(tmp_path, chunks=3, consumers=("R",))
    spec = st.make_spec(100, 3, ["R"], PRIORS, total_steps=251)
    st.atomic_json(s / "spec.json", spec)
    ready_all(s, spec)
    assert [arg(st.train_command(s, spec, "R", k, []), "--stop-at") for k in range(3)] == ["100", "200", "251"]
    monkeypatch.setattr(st, "ROOT", tmp_path)
    (tmp_path / "runs/train/R").mkdir(parents=True)
    (tmp_path / "runs/train/R/ema_step000251.pt").touch()
    evals = []
    assert st.consume(s, "R", "0", [], run=lambda *a, **k: SimpleNamespace(returncode=0),
                      eval_script="e", popen=lambda c, **k: evals.append(c[-2])) == 0
    assert evals == ["R"]
    for total in (200, 301):
        with pytest.raises(ValueError):
            st.make_spec(100, 3, ["R"], PRIORS, total_steps=total)


def test_regression_v4_default_and_disk_estimate_cli(tmp_path, monkeypatch):
    p = st.parse_prior("r2=graph:4:0.78:1:9000000")
    spec = st.make_spec(1, 1, ["a"], [p], task="regression", reg_factors="noise")
    assert "--p-binary" not in st.gen_command(spec, spec["priors"][0], tmp_path, 9000000, 1, 0)
    with pytest.raises(ValueError):
        st.make_spec(1, 1, ["a"], [dict(p, p_binary=.4)], task="regression", reg_factors="noise")
    calls = []
    monkeypatch.setattr(st, "generate", lambda *a, **kw: calls.append((a, kw)) or 0)
    assert st.main(["gen", str(tmp_path), "--gb-per-ktask", ".3", "--until", "2"]) == 0
    assert calls[0][0][-1] == .3 and calls[0][1] == {"until": 2}
    with pytest.raises(SystemExit):
        st.main(["gen", str(tmp_path), "--gb-per-ktask", "nan"])
