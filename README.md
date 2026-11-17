# FF Screener — replicá portfolios Fama-French con datos gratuitos

Screener interactivo (Streamlit) que reproduce el universo y los breakpoints de los portfolios de la
[Data Library de Ken French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
para ver qué acciones pertenecen hoy a cada celda (ej. *Small / High B/M*, *Small / Winner*) e invertir en ellas.

## Uso
```bash
pip install -r requirements.txt
export FFSCREENER_CONTACT="Tu Nombre tu@mail.com"   # la SEC exige un User-Agent identificable
python -m ffscreener.update          # descarga todo (precios ~15-25 min la primera vez)
streamlit run app.py
```

## Fuentes (todas gratuitas)
| Dato | Fuente |
|---|---|
| Breakpoints oficiales NYSE (ME, B/M, Prior 2-12) | Ken French Data Library |
| Universo, exchange (NYSE/AMEX/NASDAQ), ETFs | NASDAQ Trader symbol directory + SEC `company_tickers_exchange` |
| Book equity, acciones en circulación | SEC EDGAR XBRL *frames* |
| Precios, volumen | yfinance |

## Cómo se construye cada variable
- **ME** = acciones (portada SEC más reciente) × precio ajustado por splits al cierre de la formación.
- **B/M** = book equity del ejercicio fiscal t-1 (último cierre del año calendario t-1) / ME de diciembre t-1. Solo BE > 0.
- **Prior 2-12** = retorno total desde fin de (F−12 meses) hasta fin de (F−1 mes), en %.
- Un valor igual al breakpoint cae en el grupo inferior. Breakpoints = percentiles de acciones NYSE solamente.

## Agregar un portfolio
Editá `ffscreener/portfolios.yaml`: cada portfolio es una lista de *sorts* (`var`, `cuts` en percentiles NYSE, `labels`).
También podés armar uno desde la app ("Personalizado").

## Validación contra French (formación jun-2026, NYSE)
| | p10 | p30 | p50 | p70 | p90 |
|---|---|---|---|---|---|
| ME (propio vs French) | −3% | −8% | +1% | +1% | +1% |
| B/M | +6% | +16% | +11% | +10% | +9% |
Universo NYSE con B/M válido: 1.007 acciones (French: 1.009). El size replica bien; el B/M propio sale ~10% más alto
(causa no resuelta; sumar impuestos diferidos lo empeora). Al operar cerca de un corte, verificá la acción a mano.

## Limitaciones conocidas (datos gratuitos)
- **Sin delistings** y universo *actual*: sirve para decidir qué comprar hoy, no para backtests fieles.
- Acciones multiclase: el ME de la firma puede estar subestimado (se marca `multi_class`).
- ME histórico usa acciones actuales × precio pasado (ignora emisiones/recompras posteriores).
- Book equity = patrimonio − preferentes (sin impuestos diferidos); validá contra los percentiles de French en la pestaña *Validación*.
- Los breakpoints de French se publican con rezago; sin publicación se calculan con la muestra NYSE propia.
