#!/usr/bin/env python3
"""Ejemplos verificables del informe PC5 (pérdida y L-BFGS).

Cada función reproduce un ejemplo (A–I) del informe con aritmética exacta
(fractions.Fraction) y comprueba con `assert` los mismos números que aparecen
en el texto; numpy se usa solo para contrastes en coma flotante (diferencias
finitas, autovalores). Uso:

    python3 codigo/ejemplos_verificables.py            # todos
    python3 codigo/ejemplos_verificables.py D H I      # algunos

Fuente del plan: plan_ejemplos_verificables.md (paso 1).
"""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

C1 = Fr(1, 10**4)  # constante de Armijo
C2 = Fr(9, 10)  # constante de curvatura (Wolfe fuerte)


# ---------------------------------------------------------------------------
# Álgebra lineal exacta mínima (vectores = tuplas, matrices = listas de filas)
# ---------------------------------------------------------------------------
def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def matvec(A, v):
    return tuple(dot(fila, v) for fila in A)


def matmul(A, B):
    Bt = list(zip(*B))
    return [[dot(fila, col) for col in Bt] for fila in A]


def transpose(A):
    return [list(c) for c in zip(*A)]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def add(A, B):
    return [[a + b for a, b in zip(fa, fb)] for fa, fb in zip(A, B)]


def scal(c, A):
    return [[c * a for a in fila] for fila in A]


def identidad(n):
    return [[Fr(int(i == j)) for j in range(n)] for i in range(n)]


def det2(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def vsum(u, v, c=Fr(1)):
    """u + c·v."""
    return tuple(a + c * b for a, b in zip(u, v))


def bfgs_inversa(H, s, y):
    """H⁺ = Vᵀ H V + ρ s sᵀ, con ρ = 1/(sᵀy), V = I − ρ y sᵀ."""
    rho = 1 / dot(s, y)
    V = add(identidad(len(s)), scal(-rho, outer(y, s)))
    return add(matmul(transpose(V), matmul(H, V)), scal(rho, outer(s, s)))


def dos_lazos(g, pares, gamma):
    """Recursión de dos lazos (Algoritmo alg:lbfgs-dos-lazos): r = H g."""
    q = tuple(g)
    a = []
    for s, y in reversed(pares):  # i = k−1, …, k−ℓ
        rho = 1 / dot(s, y)
        ai = rho * dot(s, q)
        q = vsum(q, y, -ai)
        a.append(ai)
    r = tuple(gamma * qi for qi in q)
    for (s, y), ai in zip(pares, reversed(a)):  # i = k−ℓ, …, k−1
        rho = 1 / dot(s, y)
        b = rho * dot(y, r)
        r = vsum(r, s, ai - b)
    return r


def fmt(x):
    """Fracción con su valor decimal."""
    if isinstance(x, Fr):
        return f"{x}" if x.denominator == 1 else f"{x} (≈{float(x):.4f})"
    return f"{x:.4f}"


def fvec(v):
    return "(" + ", ".join(str(x) for x in v) + ")"


def fmat(A):
    return "[" + ", ".join(fvec(f) for f in A) + "]"


def titulo(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------
# Pérdida de entropía cruzada binaria
# ---------------------------------------------------------------------------
def perdida(y, p):
    """f(y, p) = −y log p − (1−y) log(1−p), con y ∈ {0, 1}."""
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def ejemplo_A():
    titulo("Ejemplo A: pérdida individual")
    p = 0.8
    f1, f0 = perdida(1, p), perdida(0, p)
    print(f"y=1, p=0.8: f = −log 0.8 = {f1:.4f}")
    print(f"y=0, p=0.8: f = −log 0.2 = {f0:.4f}")
    assert abs(f1 - 0.2231) < 5e-5 and abs(f0 - 1.6094) < 5e-5
    for y in (0, 1):
        for q in (0.05, 0.3, 0.5, 0.8, 0.99):
            assert abs(perdida(y, q) - (-math.log(1 - abs(q - y)))) < 1e-12
    print("f(y,p) = −log(1 − |p − y|) para y∈{0,1}: comprobado.")
    print(f"Asimetría: con el mismo p, la etiqueta errónea cuesta {f0 / f1:.2f} veces más.")


def ejemplo_B():
    titulo("Ejemplo B: pérdida empírica con 3 vértices")
    y = (1, 0, 1)
    p = (0.9, 0.2, 0.6)
    F = sum(perdida(yi, pi) for yi, pi in zip(y, p)) / 3
    print(f"y={y}, p={p}: F = {F:.4f}")
    for i, (yi, pi) in enumerate(zip(y, p), start=1):
        cota = -math.log(1 - abs(pi - yi)) / 3
        print(f"  vértice {i}: F ≥ (1/3)(−log(1−|p_i−y_i|)) = {cota:.4f}")
        assert F >= cota > 0
    print("Basta un vértice con p_i ≠ y_i para que F > 0.")


def ejemplo_C():
    titulo("Ejemplo C: regresión logística, gradiente y constante L")
    x = np.array([1.0, 2.0])
    y = 1.0

    def f(w, b):
        p = 1 / (1 + math.exp(-(w @ x + b)))
        return perdida(y, p)

    w, b = np.zeros(2), 0.0
    p = 0.5
    grad = (p - y) * x
    print(f"w=0, b=0, x=(1,2), y=1: p=1/2, ∇_w f = (p−y)x = {tuple(grad.tolist())}")
    assert np.allclose(grad, [-0.5, -1.0])
    h = 1e-6
    dif = np.array([(f(w + h * e, b) - f(w - h * e, b)) / (2 * h) for e in np.eye(2)])
    print(f"diferencias centradas: {np.round(dif, 8)}")
    assert np.allclose(dif, grad, atol=1e-8)
    # Hessiano en w: p(1−p) x xᵀ; p(1−p) ≤ 1/4 con igualdad en p = 1/2.
    L = Fr(1, 4) * dot((1, 2), (1, 2))
    lam = np.linalg.eigvalsh(0.25 * np.outer(x, x)).max()
    print(f"λ_max(p(1−p)xxᵀ) ≤ ‖x‖²/4 = {L}; en p=1/2 se alcanza: {lam:.4f}")
    assert L == Fr(5, 4) and abs(lam - 1.25) < 1e-12


# ---------------------------------------------------------------------------
# Cuadrática F(θ) = ½ θᵀ A θ, A = diag(1, 4)
# ---------------------------------------------------------------------------
A = [[Fr(1), Fr(0)], [Fr(0), Fr(4)]]


def Fq(t):
    return dot(t, matvec(A, t)) / 2


def gq(t):
    return matvec(A, t)


def wolfe_fuerte(t, d, alpha):
    """Devuelve (armijo, curvatura) para el paso alpha."""
    g0d = dot(gq(t), d)
    t1 = vsum(t, d, alpha)
    armijo = Fq(t1) <= Fq(t) + C1 * alpha * g0d
    curv = abs(dot(gq(t1), d)) <= C2 * abs(g0d)
    return armijo, curv


def ejemplo_D():
    titulo("Ejemplo D: condiciones de Wolfe fuertes")
    t0 = (Fr(1), Fr(1))
    d0 = tuple(-gi for gi in gq(t0))
    assert d0 == (-1, -4) and Fq(t0) == Fr(5, 2) and dot(gq(t0), d0) == -17
    print(f"θ0={fvec(t0)}, F(θ0)=5/2, d0={fvec(d0)}, g0ᵀd0=−17, c1=1e-4, c2=0.9")
    esperado = {
        Fr(1): (Fr(18), False, False),
        Fr(1, 100): (Fr(9333, 4000), True, False),
        Fr(1, 4): (Fr(9, 32), True, True),
    }
    for alpha, (Fesp, arm_esp, curv_esp) in esperado.items():
        F1 = Fq(vsum(t0, d0, alpha))
        arm, curv = wolfe_fuerte(t0, d0, alpha)
        print(f"  α={alpha}: F={fmt(F1)}, Armijo {'sí' if arm else 'no'}, "
              f"curvatura {'sí' if curv else 'no'}")
        assert (F1, arm, curv) == (Fesp, arm_esp, curv_esp)
    # Paso exacto: φ'(α) = gᵀ(θ0+αd)·d = 0
    a_opt = -dot(gq(t0), d0) / dot(d0, matvec(A, d0))
    print(f"  paso exacto α* = {fmt(a_opt)}")
    assert a_opt == Fr(17, 65) and wolfe_fuerte(t0, d0, a_opt) == (True, True)
    # φ(α) = 5/2 − 17α + (65/2)α², φ'(α) = −17 + 65α: los pasos de Wolfe fuerte
    # forman [17/650, 323/650]; Armijo vale hasta α ≤ 2(17 − 17c1)/65.
    lo, hi = Fr(17, 650), Fr(323, 650)
    assert wolfe_fuerte(t0, d0, lo) == (True, True) and wolfe_fuerte(t0, d0, hi) == (True, True)
    eps = Fr(1, 10**6)
    assert not wolfe_fuerte(t0, d0, lo - eps)[1] and not wolfe_fuerte(t0, d0, hi + eps)[1]
    assert 2 * 17 * (1 - C1) / 65 > hi
    print(f"  pasos de Wolfe fuerte: α ∈ [17/650, 323/650] ≈ [{float(lo):.4f}, {float(hi):.4f}]")


def ejemplo_E():
    titulo("Ejemplo E: ángulo con la dirección de máximo descenso")
    H = [[Fr(1), Fr(0)], [Fr(0), Fr(1, 4)]]
    g = (Fr(1), Fr(1))
    d = tuple(-x for x in matvec(H, g))
    num = -dot(g, d)
    cos2 = num**2 / (dot(g, g) * dot(d, d))  # cos²ψ exacto
    assert num == Fr(5, 4) and cos2 == Fr(25, 34)
    cos = math.sqrt(cos2)
    kappa = 4
    print(f"H=diag(1,1/4), κ(H)={kappa}, g=(1,1), d=−Hg={fvec(d)}")
    print(f"cos ψ = 5/√34 ≈ {cos:.4f} ≥ 1/κ = {1 / kappa}  (cota válida, no ajustada)")
    assert abs(cos - 0.857) < 5e-4 and cos >= 1 / kappa


def ejemplo_F():
    titulo("Ejemplo F: actualización BFGS de la inversa")
    s, y = (Fr(1), Fr(0)), (Fr(2), Fr(1))
    Hp = bfgs_inversa(identidad(2), s, y)
    print(f"s={fvec(s)}, y={fvec(y)}, sᵀy={dot(s, y)}, H=I ⇒ H⁺={fmat(Hp)}")
    assert Hp == [[Fr(3, 4), Fr(-1, 2)], [Fr(-1, 2), Fr(1)]]
    assert matvec(Hp, y) == s and det2(Hp) == Fr(1, 2) and Hp[0][0] > 0
    print(f"ecuación secante H⁺y = s: sí; det H⁺ = {det2(Hp)} > 0 y H⁺₁₁ > 0 ⇒ H⁺ ≻ 0")


def ejemplo_G():
    titulo("Contraejemplo G: sᵀy < 0 destruye la definición positiva")
    s, y = (Fr(1), Fr(0)), (Fr(-1), Fr(1))
    Hp = bfgs_inversa(identidad(2), s, y)
    print(f"s={fvec(s)}, y={fvec(y)}, sᵀy={dot(s, y)} ⇒ H⁺={fmat(Hp)}, det={det2(Hp)}")
    assert dot(s, y) == -1 and Hp == [[0, 1], [1, 1]] and det2(Hp) == -1
    # Versión 1D: F(θ) = −θ²/2, g(θ) = −θ
    th0, dd, alpha = Fr(1), Fr(1), Fr(1)
    g = lambda th: -th  # noqa: E731
    th1 = th0 + alpha * dd
    s1, y1 = th1 - th0, g(th1) - g(th0)
    print(f"1D: F=−θ²/2, θ0=1, d=1 (g0·d=−1<0), α=1 ⇒ sᵀy = {s1 * y1}")
    assert s1 * y1 == -1
    curv_debil = g(th1) * dd >= C2 * g(th0) * dd
    print(f"  curvatura: g1·d = {g(th1) * dd} ≥ c2·g0·d = {fmt(C2 * g(th0) * dd)}? "
          f"{'sí' if curv_debil else 'no'} ⇒ Wolfe rechaza el paso")
    assert not curv_debil


def ejemplo_H():
    titulo("Ejemplo H: recursión de dos lazos con M = 1")
    s, y = (Fr(1), Fr(0)), (Fr(2), Fr(1))
    gamma = dot(s, y) / dot(y, y)
    g = (Fr(1), Fr(1))
    r = dos_lazos(g, [(s, y)], gamma)
    H = bfgs_inversa(scal(gamma, identidad(2)), s, y)
    print(f"par (s,y)=({fvec(s)},{fvec(y)}), γ = sᵀy/yᵀy = {gamma}")
    print(f"dos lazos: r = {fvec(r)};  H explícita = {fmat(H)}, H g = {fvec(matvec(H, g))}")
    assert gamma == Fr(2, 5) and r == (Fr(2, 5), Fr(1, 5))
    assert H == [[Fr(3, 5), Fr(-1, 5)], [Fr(-1, 5), Fr(2, 5)]] and matvec(H, g) == r


def ejemplo_I():
    titulo("Ejemplo I: máximo descenso frente a L-BFGS (M = 1)")
    t0 = (Fr(4), Fr(1))
    F0 = Fq(t0)

    def paso_exacto(t, d):
        return -dot(gq(t), d) / dot(d, matvec(A, d))

    # Máximo descenso con búsqueda exacta
    t = t0
    print(f"θ0={fvec(t0)}, F(θ0)={F0}")
    for k in range(6):
        d = tuple(-x for x in gq(t))
        a = paso_exacto(t, d)
        t_new = vsum(t, d, a)
        ratio = Fq(t_new) / Fq(t)
        if k == 0:
            assert a == Fr(2, 5) and t_new == (Fr(12, 5), Fr(-3, 5))
            print(f"  MD k=0: α0={a}, θ1={fvec(t_new)}")
        assert ratio == Fr(9, 25) and t_new != (0, 0)
        t = t_new
    print("  MD: F_(k+1)/F_k = 9/25 = ((κ−1)/(κ+1))² en las 6 iteraciones; θ_k ≠ 0 (zigzag)")

    # L-BFGS: primer paso = máximo descenso exacto; luego M = 1
    t1 = (Fr(12, 5), Fr(-3, 5))
    s0, y0 = vsum(t1, t0, -1), vsum(gq(t1), gq(t0), -1)
    gamma = dot(s0, y0) / dot(y0, y0)
    d1 = tuple(-x for x in dos_lazos(gq(t1), [(s0, y0)], gamma))
    a1 = paso_exacto(t1, d1)
    t2 = vsum(t1, d1, a1)
    print(f"  L-BFGS: s0={fvec(s0)}, y0={fvec(y0)}, γ={gamma}")
    print(f"          d1={fvec(d1)}, α1={a1} ⇒ θ2={fvec(t2)}")
    assert gamma == Fr(5, 17) and d1 == (Fr(-96, 85), Fr(24, 85))
    assert a1 == Fr(17, 8) and t2 == (0, 0)
    assert dot(s0, matvec(A, d1)) == 0  # d1 es A-conjugada de s0 (y de d0)
    # MD en forma cerrada: θ_k = (3/5)^k (4, (−1)^k)
    t = t0
    for k in range(1, 7):
        d = tuple(-x for x in gq(t))
        t = vsum(t, d, paso_exacto(t, d))
        assert t == (Fr(3, 5) ** k * 4, Fr(3, 5) ** k * (-1) ** k)
    print("  d1 es A-conjugada de s0 (s0ᵀA d1 = 0); MD: θ_k = (3/5)^k (4, (−1)^k).")
    assert wolfe_fuerte(t1, d1, a1) == (True, True)
    print("  L-BFGS alcanza el minimizador θ* = 0 en dos pasos; α1 cumple Wolfe fuerte.")


def puente_elliptic():
    titulo("Puente con Elliptic (resultados/resultados_elliptic_lbfgs.json)")
    ruta = Path(__file__).resolve().parent.parent / "resultados" / "resultados_elliptic_lbfgs.json"
    if not ruta.exists():
        print(f"(no se encontró {ruta}; se omite)")
        return
    datos = json.loads(ruta.read_text())
    err = datos["gradient_checks_max_rel_err"]
    print("error relativo máx. del gradiente (diferencias finitas): "
          + ", ".join(f"{k}={v:.1e}" for k, v in err.items()))
    for nombre, run in datos["runs"].items():
        if run.get("method") != "lbfgs":
            continue
        print(f"  {nombre}: {run['iterations']} it., reinicios={run['memory_resets']}, "
              f"fallos Wolfe={run['wolfe_fallbacks']}, cos ψ mín={run['cos_psi_min']:.3f}, "
              f"parada: {run['stop_reason']}")


EJEMPLOS = {
    "A": ejemplo_A, "B": ejemplo_B, "C": ejemplo_C, "D": ejemplo_D, "E": ejemplo_E,
    "F": ejemplo_F, "G": ejemplo_G, "H": ejemplo_H, "I": ejemplo_I, "P": puente_elliptic,
}


def main(argv):
    elegidos = [a.upper() for a in argv] or list(EJEMPLOS)
    for clave in elegidos:
        EJEMPLOS[clave]()
    print("\nTodas las comprobaciones pasaron.")


if __name__ == "__main__":
    main(sys.argv[1:])
