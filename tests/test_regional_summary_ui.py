import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_regional_summary_ui_is_not_rendered_but_helpers_are_retained():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    regional = _function(tree, "render_regional_analysis")
    called_names = {
        node.func.id
        for node in ast.walk(regional)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "render_regional_summary" not in called_names
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "render_regional_summary"
        for node in tree.body
    )
    assert "render_national_summary" in {
        node.func.id
        for node in ast.walk(_function(tree, "render_national_overview"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_weekly_kpis_are_the_first_regional_analysis_subheader():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    regional = _function(tree, "render_regional_analysis")
    subheaders = [
        node.args[0].value
        for node in ast.walk(regional)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "subheader"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]

    assert subheaders[0] == "주간 핵심지표 비교"
    assert not any("핵심 요약" in heading for heading in subheaders)
