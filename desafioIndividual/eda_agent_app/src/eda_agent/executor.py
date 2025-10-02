from __future__ import annotations
import io, contextlib, ast
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Builtins seguros e suficientes para Pandas/Numpy/Matplotlib
SAFE_BUILTINS = {
    # tipos e checagens
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    "object": object,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,

    # iteração e utilidades
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,

    # I/O limitado
    "print": print,

    # exceções comuns (para raise/except funcionarem)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AssertionError": AssertionError,
    "ZeroDivisionError": ZeroDivisionError,
}

class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node):  # bloqueia qualquer import
        raise ValueError("Uso de 'import' não permitido no código gerado.")
    def visit_ImportFrom(self, node):
        raise ValueError("Uso de 'from ... import' não permitido no código gerado.")
    def visit_Attribute(self, node):
        # bloqueia acesso a atributos dunder (ex.: __class__, __dict__, etc.)
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError("Acesso a atributos internos não permitido.")
        self.generic_visit(node)
    def visit_Name(self, node):
        # bloqueia nomes internos críticos
        if node.id in {"__builtins__", "__loader__", "__package__", "__spec__"}:
            raise ValueError("Nome interno bloqueado.")
    def visit_Call(self, node):
        # funções perigosas
        banned = {"eval","exec","open","compile","input","__import__","globals","locals","vars","dir","getattr","setattr","delattr"}
        if isinstance(node.func, ast.Name) and node.func.id in banned:
            raise ValueError(f"Chamada a função proibida: {node.func.id}")
        self.generic_visit(node)

def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def run_generated_code(code: str, extra_globals: Dict[str, Any]) -> Dict[str, Any]:
    # validação AST
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Erro de sintaxe no código gerado: {e}")
    _SafetyVisitor().visit(tree)

    # ambiente de execução com builtins restritos
    sandbox_globals = {"__builtins__": SAFE_BUILTINS}
    sandbox_globals.update(extra_globals)

    # executa capturando stdout
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        exec(compile(tree, filename="<llm_code>", mode="exec"), sandbox_globals, sandbox_globals)

    text = sandbox_globals.get("RESULT_TEXT")
    stdout = f.getvalue()

    # captura todas as figuras abertas
    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        images.append(_fig_to_png_bytes(fig))
    plt.close("all")

    return {
        "stdout": stdout,
        "text": text if isinstance(text, str) else "",
        "images": images,
    }
