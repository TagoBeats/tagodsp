# True Peak Limiter (dynamics/true_peak_limiter.py)

Haelt eine in dBTP gesetzte Decke ein, nicht nur eine Sample-Peak-Decke. Zweite
Haelfte des TagoClip-Pro-Versprechens: der Clipper darf so hart zupacken wie
gewollt, die Datei besteht die Plattform-Pruefung trotzdem.

## Warum das gebraucht wird

Gemessen am 21.09.2026 (`docs/measurements/isp_sweep_2026-09-21.md`): nach dem
Clipping liegt der True Peak bis zu **5,8 dB ueber der eingestellten Decke**,
und selbst mit achtfachem Oversampling bleiben **4,2 dB**. Oversampling
verschiebt das Problem in die Sichtbarkeit, geloest wird es nie. Ohne eine
Stufe, die den True Peak selbst kennt, haelt keine Einstellung des Plugins das,
was ihr Regler verspricht.

## Aufbau, und warum die Reihenfolge nicht beliebig ist

1. **Oversampeln**, damit die Zwischenwerte ueberhaupt sichtbar sind. Ein
   Detektor, der nur die Samples sieht, kann Inter-Sample-Peaks nicht kennen.
2. **Benoetigte Verstaerkung je Sample**, `g_req = min(1, ceiling / |x|)`.
3. **Laufendes Minimum** ueber ein Fenster der Laenge `L` (das Lookahead).
4. **Glaetten** mit einem Hann-Kernel, dessen Traeger hoechstens `L/2` breit ist.

Schritt 3 vor Schritt 4 ist der ganze Trick, und die Garantie haengt daran:
nach dem laufenden Minimum gilt `g_env[m] <= g_req[n]` fuer jedes `m` im
Abstand `L/2` um `n`. Der Glaettungskernel greift nur auf genau diese Nachbarn
zu und summiert sich zu 1, also ist auch der geglaettete Wert nicht groesser als
`g_req[n]`. Die Decke kann damit nicht gerissen werden.

Andersherum (erst glaetten, dann Minimum) waere die Ordnung dahin: der
geglaettete Wert darf dann ueber der noetigen Verstaerkung liegen, und genau an
der Transiente, wo es drauf ankommt, tut er das auch.

## Was das kostet

Das Lookahead ist echte Latenz, `L/2` Samples. Ein Limiter ohne Lookahead
muesste entweder nachlaufen (dann ist der Peak schon raus) oder so schnell
zupacken, dass es verzerrt.

Release ist in dieser Fassung das Glaettungsfenster selbst, es gibt keine
getrennte Release-Zeit. Das ist bewusst: jede zusaetzliche Stufe muss die
Verstaerkung **nur weiter senken** duerfen, sonst faellt die Garantie. Eine
eigene Release-Kurve ist ein spaeterer Feinschliff, kein Prototyp-Thema.

## Grenzen

- Offline, ganzer Buffer. Der Detektor-Resampler hat keinen Blockzustand.
- Stereo laeuft **gekoppelt**: die Verstaerkung wird aus dem lautesten Kanal
  gebildet und auf alle angewendet, sonst wandert das Stereobild bei jedem
  Eingriff.
- Die Verstaerkung wird auf der Basisrate angewendet, gepruefte Decke ist die
  am Ausgang gemessene. Die Einhaltung wird gemessen und nicht aus der
  Konstruktion geschlossen.

## Warum es einen Sicherheitsabstand gibt

Die Garantie oben gilt fuer das, was der Detektor **sieht**. Ein Detektor mit
endlichem Oversampling sieht aber nicht alles: das Maximum kann zwischen zwei
seiner eigenen Stuetzstellen liegen, genau derselbe blinde Fleck wie bei jedem
True-Peak-Meter.

Gemessen am 21.09.2026 an geclipptem Rauschen, Ziel -1,0 dBTP, gegengeprueft
mit einem 32x-Meter:

| Detektor | gemessenes Ergebnis |
| :--- | :--- |
| 4x | -0,74 dBTP |
| 8x | -0,91 dBTP |
| 16x | -0,98 dBTP |

Wiederholte Durchlaeufe aendern daran nichts, das Lookahead auch nicht. Es ist
keine Seitenbandwirkung der zeitvariablen Verstaerkung, wie hier zuerst
vermutet, sondern schlicht Detektor-Aufloesung.

Den Detektor genau so fein zu machen wie das eigene Messgeraet waere eine
Optimierung auf genau dieses eine Instrument. Ein feineres Meter findet danach
wieder etwas. Deshalb zielt der Limiter um `safety_db` **unter** die gesetzte
Decke, per Default 0,05 dB. Das kostet Lautheit, die niemand hoert, und kauft
dafuer die Zusage: die Zahl am Regler wird auch von einem feineren Messgeraet
nicht ueberschritten.
