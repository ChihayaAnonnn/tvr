import multiprocessing
import pytest


def test_gpu_budget_counts_active_reservations_all_cards_and_terminal_time():
    from semantic_retrieval.full_data.controller import budget_projection
    runs = [dict(category='software', gpu_count=4, max_wall_seconds=1800, state='running', elapsed_seconds=None),
            dict(category='software', gpu_count=2, max_wall_seconds=7200, state='completed', elapsed_seconds=3600),
            dict(category='scientific', gpu_count=4, max_wall_seconds=86400, state='reserved', elapsed_seconds=None)]
    assert budget_projection(runs) == dict(software=4.0, scientific=96.0)
    with pytest.raises(ValueError):
        budget_projection([dict(runs[0], state='failed', elapsed_seconds=None)])
    with pytest.raises(ValueError):
        budget_projection([dict(runs[1], elapsed_seconds=float('nan'))])


def test_successor_cap_is_executable_and_rejects_invalid_accounting():
    from semantic_retrieval.full_data.controller import check_successors
    check_successors(0, 2); check_successors(2, 2)
    for attempts in (3, -1, True):
        with pytest.raises(ValueError): check_successors(attempts, 2)


def test_gpu_worker_rejects_direct_launch_without_budget_controller(tmp_path):
    from semantic_retrieval.full_data.controller import validate_controller_launch
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_controller_launch(tmp_path, tmp_path, dict(environment=dict(gpu_count=4)))


def _hold(path, ready, release):
    from semantic_retrieval.full_data.controller import campaign_lock
    with campaign_lock(path):
        ready.set(); release.wait(10)


def test_active_campaign_lock_is_atomic_between_real_processes(tmp_path):
    from semantic_retrieval.full_data.controller import campaign_lock
    ctx = multiprocessing.get_context('spawn'); ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold, args=(tmp_path/'campaign.lock', ready, release)); process.start()
    try:
        assert ready.wait(8)
        with pytest.raises(RuntimeError, match='active'):
            with campaign_lock(tmp_path/'campaign.lock'): pass
    finally:
        release.set(); process.join(10)
    assert process.exitcode == 0
    with campaign_lock(tmp_path/'campaign.lock'): pass
