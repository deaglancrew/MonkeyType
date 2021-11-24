from pandera_example import makes_df, takes_df
from monkeytype import trace
from monkeytype_config import CONFIG
from monkeytype.cli import MonkeyTypeError, build_module_stubs_from_traces
from tuple_example import takes_tuples, creates_tuple, takes_tuple_returns_int
import os


def print_stub():
    module, qualname = ('pandera_example', 'takes_df')
    thunks = CONFIG.trace_store().filter(module, qualname)
    traces = []
    failed_to_decode_count = 0
    for thunk in thunks:
        try:
            traces.append(thunk.to_trace())
        except MonkeyTypeError as mte:
            print(f'WARNING: Failed decoding trace: {mte}')
            failed_to_decode_count += 1
    if failed_to_decode_count:
        print(f'{failed_to_decode_count} traces failed to decode; use -v for details')
    if not traces:
        return None
    rewriter = CONFIG.type_rewriter()
    stubs = build_module_stubs_from_traces(
        traces,
        CONFIG.type_getter(),
        rewriter=rewriter,
    )
    print(stubs.get(module, None).render())


if __name__ == '__main__':
    os.remove('monkeytype.sqlite3')
    with trace(CONFIG):
        takes_tuple_returns_int(creates_tuple())
        takes_tuples(list(creates_tuple() for _ in range(4)))
        takes_df(makes_df())
    print_stub()
