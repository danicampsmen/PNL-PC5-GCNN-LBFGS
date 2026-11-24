# Redes neuronales convolucionales sobre grafos

Práctica Calificada 5 de Programación No Lineal (CM4E2), ciclo 2026-II,
Universidad Nacional de Ingeniería. Profesor: Dr. Ernesto Oré Albornoz.

Integrantes: Francisco Alonso Basurto Pompilla, David Daniel Campos Mendoza y
Martin Rojas Medrano.

El trabajo estudia una red neuronal convolucional sobre grafos (GCNN) y la
entrena con L-BFGS y búsqueda lineal de Wolfe fuerte sobre el conjunto de datos
Elliptic, que contiene transacciones de bitcoin.

## Contenido

- `informe/`: informe final. Incluye la fuente `PNL_Indice_Integrado.tex`, el
  PDF compilado, la bibliografía y las figuras.
- `codigo/`:
  - `gcn_elliptic_lbfgs.py`: L-BFGS (Andrei, 2020, Algoritmo 1.6) y el método
    del gradiente con Wolfe sobre Elliptic.
  - `gcn_elliptic_nivel2.py`: régimen convexo ($L_{\mathrm{net}}=1$), GCNN con
    separación ego-vecino ($L_{\mathrm{net}}=2$) y *multi-start*.
  - `ejemplos_verificables.py`: comprobación con aritmética exacta de los
    ejemplos numéricos del informe.
- `resultados/`: resúmenes en JSON y el historial de cada iteración en CSV
  (con topes de 300 y 1000 iteraciones).

## Compilar el informe

```sh
cd informe
lualatex --shell-escape PNL_Indice_Integrado.tex
biber PNL_Indice_Integrado
lualatex --shell-escape PNL_Indice_Integrado.tex
lualatex --shell-escape PNL_Indice_Integrado.tex
```

## Reproducir los experimentos

Se necesitan Python 3 con `numpy`, `scipy` y `matplotlib`.

El conjunto de datos Elliptic no se incluye en el repositorio. El programa lee
`elliptic_txs_classes.csv`, `elliptic_txs_features.csv` y
`elliptic_txs_edgelist.csv` del archivo `Fundamentos_Matematicos_GNN.zip`, que
debe estar en la raíz del repositorio. `gcn_elliptic_lbfgs.py` también acepta
otra ruta con `--zip-path` o un directorio con los CSV ya extraídos con
`--data-dir`.

```sh
python3 codigo/gcn_elliptic_lbfgs.py --max-iter 300
python3 codigo/gcn_elliptic_lbfgs.py --max-iter 1000 --sufijo _1000
python3 codigo/gcn_elliptic_nivel2.py
python3 codigo/ejemplos_verificables.py
```

Las salidas se escriben en `resultados/` y `figuras/`.
