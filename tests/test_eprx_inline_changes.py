import ast
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


SOURCE = Path("app.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class TableTarget:
    def markdown(self, content, **kwargs):
        self.html = content


@pytest.mark.parametrize("rate,text", [
    (.0528, "(+5.28%)"), (-.0528, "(-5.28%)"), (0., "(+0.00%)"),
    (None, None), (np.nan, None), (np.inf, None), (-np.inf, None),
])
def test_inline_format_and_invalid_rate_omission(rate, text):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "render_hierarchical_metric_table")
    namespace = {"pd": pd, "np": np, "escape": escape}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    target = TableTarget()
    namespace[node.name](target, pd.DataFrame({"도쿄": ["444.22"]}, index=["평균 모집량"]),
                         change_rates={("평균 모집량", "도쿄"): rate})
    assert "444.22" in target.html
    assert ".metric-current-value { color: inherit; }" in target.html
    assert "color: #8a8a8a; font-size: 0.85em" in target.html
    assert ".metric-inline-value { white-space: nowrap; }" in target.html
    if text:
        assert f'class="metric-change-rate">{text}</span>' in target.html
    else:
        assert 'class="metric-change-rate"' not in target.html
        assert '<td class="metric-value-cell">444.22</td>' in target.html


@pytest.mark.parametrize("previous", [None, 0])
def test_runtime_missing_or_zero_previous_week_keeps_current_only(previous):
    nodes = [n for n in TREE.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    script = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    script += '\nEPRX_REGION_OPTIONS = EPRX_AREA_OPTIONS\n'
    script += '''
rows = []
for date, amount in [('2026-07-20', 100), ('2026-07-13', PREVIOUS)]:
    if amount is None:
        continue
    for area in EPRX_AREA_OPTIONS.values():
        rows.append(dict(delivery_date=date, area=area, frequency_zone='50Hz',
            period_no=1, period_start='00:00', procurement_volume=amount,
            bid_volume=amount, awarded_volume=amount,
            max_price=10, min_price=2, avg_price=5))
data = add_week_columns(pd.DataFrame(rows))
render_regional_analysis(st, data, pd.Timestamp('2026-07-20'), 1, '엔')
'''.replace("PREVIOUS", repr(previous))
    app = AppTest.from_string(script, default_timeout=30).run()
    app.radio(key="regional_view_mode").set_value("도쿄").run()
    assert not app.exception
    table = next(m.value for m in app.markdown if '<table' in m.value)
    assert table.count('<td class="metric-value-cell">100.00</td>') == 3
    assert 'class="metric-change-rate"' not in table
    assert "100.00%" in table
    assert "전주 대비 변화" not in [s.value for s in app.subheader]


def test_comparison_not_recalculated_by_ui():
    function = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "render_regional_analysis")
    body = ast.get_source_segment(SOURCE, function)
    assert body.count("calculate_previous_week_comparison(") == 1
    assert 'previous["지역"].eq(view) & previous["지표"].eq(metric), "변화율"' in body
    assert 'target.subheader("전주 대비 변화")' not in body
