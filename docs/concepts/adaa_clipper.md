# ADAA Clipper (distortion/adaa.py)

Antiderivative Antialiasing erster Ordnung fuer die Clipping-Kurven aus
`distortion/clipper.py`. Ziel ist Aliasunterdrueckung ohne den Preis von hohem
Oversampling: weniger Latenz, weniger Rechenlast, gleicher Klangcharakter.

## Verfahren

Eine gedaechtnislose Nichtlinearitaet `f` erzeugt Obertoene oberhalb Nyquist,
die zurueckfalten. Statt das Signal hochzutasten mittelt ADAA die Kennlinie
ueber das Intervall zwischen zwei aufeinanderfolgenden Samples, indem es die
Stammfunktion `F1` auswertet:

    y[n] = (F1(x[n]) - F1(x[n-1])) / (x[n] - x[n-1])

Das ist exakt das, was eine stueckweise lineare Rekonstruktion des Eingangs
durch die Kennlinie geschickt und wieder bandbegrenzt ergeben wuerde. Es kostet
eine halbe Sample Verzoegerung.

Quelle: Parker, Zavalishin, Le Bivic, "Reducing the Aliasing of Nonlinear
Waveshaping Using Continuous-Time Convolution", DAFx-16.

## Stammfunktionen

Alle drei Kurven sind ungerade, also ist `F1` gerade und wird auf `|x|`
gerechnet. Mit `a = 1 - t` und `u = (|x| - t) / a`, unterhalb des Knies immer
`F1 = x^2 / 2`:

| Kurve | `F1` oberhalb des Knies |
| :--- | :--- |
| `hard` | `t*|x| - t^2/2` |
| `fl` | `t^2/2 + (|x| - t) - a^2 * (1 - exp(-u))` |
| `tanh` | `t^2/2 + t*(|x| - t) + a^2 * ln cosh(u)` |

Alle drei sind am Knie stetig, alle liefern bei `|x| = t` den Wert `t^2/2`.

**Das ist der Punkt, an dem dieses Projekt technisch haengt.** Die
Waveshaper-Marktrecherche stuft ADAA als akademisches Neuland ein, weil eine
frei gezeichnete Spline-Kennlinie keine geschlossene Stammfunktion besitzt und
numerisch ueber Lookup-Tables integriert werden muesste. Fuer diese drei Kurven
gilt das nicht, sie sind geschlossene Ausdruecke und ihre Stammfunktionen stehen
oben. Der Aufwand liegt damit bei Lehrbuch, nicht bei Forschung.

`ln cosh(u)` laeuft fuer grosse `u` ueber, weil `cosh` exponentiell waechst.
Gerechnet wird deshalb `u + log1p(exp(-2u)) - ln 2`, was fuer `u >= 0` stabil
ist. Ohne diese Form liefert die `tanh`-Kurve bei hohem Drive `inf`.

## Die schlecht konditionierte Stelle

Der Quotient dividiert durch `x[n] - x[n-1]`. Wird die Differenz klein, loeschen
sich im Zaehler die fuehrenden Stellen aus und der Quotient wird Rauschen. Bei
leisem oder tieffrequentem Material ist das der Normalfall, nicht die Ausnahme.

Fallback unterhalb einer Schwelle `eps` auf die direkte Auswertung in der Mitte
des Intervalls, `f((x[n] + x[n-1]) / 2)`. Das ist der Grenzwert des Quotienten
fuer verschwindende Differenz, der Uebergang ist also stetig. `eps` ist ein
Dataclass-Feld und wird gemessen, nicht geraten.

## Grenzen des Prototyps

- Offline. `process()` haelt den letzten Eingangswert als Zustand und ist damit
  blockweise aufrufbar, aber nur bei `oversample = 1`. Mit Oversampling fehlt
  dem Resampler sein Zustand, genauso wie bei `Clipper`. Das ist ein Thema fuer
  die C++-Portierung, nicht fuer den Prototyp.
- Der Zustand ist das getriebene Eingangssample, nicht das rohe. Drive gehoert
  vor die Kennlinie.
- Parameter-Automation ist ungeprueft: die Stammfunktion gilt fuer feste
  Parameter. Aendert sich Threshold oder Drive zwischen zwei Samples, passen
  `F1(x[n])` und `F1(x[n-1])` nicht mehr zusammen. Steht als Messpunkt an.
