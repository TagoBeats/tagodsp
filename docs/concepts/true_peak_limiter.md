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

0. **Tiefpass**, 0.875x Nyquist, 255 Taps, Kaiser Beta 8, relativ zu Nyquist
   definiert (also bei jeder Samplerate dieselben Taps). Gehoert zur Decke,
   nicht davor: das Signal, das erkannt wird, ist dasselbe, das rauskommt.
   Siehe "Der Fund vom 23./24.09.2026" unten, dieser Schritt ist der Fix.
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
wieder etwas. Deshalb zielt der Limiter **unter** die gesetzte Decke.

Die Tabelle, die hier stand (ein Abstand pro Oversampling-Faktor, 0,70 dB bis
0,02 dB), war falsch. Sie wurde mit einem 32x-Meter gegengeprueft, das
denselben Resampling-Filter benutzte wie der damalige Standard-Detektor
(halbe Laenge 10, Beta 12) und deshalb dieselbe Schwaeche hatte: ein endlicher
Filter mit Cutoff bei Nyquist liest Energie direkt unter Nyquist zu niedrig,
egal wie stark man oversampelt, und ein laengerer Filter derselben Bauart
konvergiert dabei viel zu langsam (256 Taps/Phase lasen bei der Nachmessung
immer noch 0,21 dB zu hoch). Meter und Detektor haben sich also gegenseitig
bestaetigt, nicht die Realitaet.

## Der Fund vom 23./24.09.2026

Gemessen mit einem idealen Meter (steiles Kaiser-14-Fenster, 32x, 512
Taps/Phase, Raender ausgeschlossen): der 16x-Detektor ohne Tiefpass liess auf
echten okayes-Stems bis zu **+1,04 dB** ueber der Decke durch, auf Rauschen bis
zu **+3,4 dB**. Ein Oracle-Detektor (unendliche Aufloesung) zeigte: das Problem
ist reine Detektor-Aufloesung nahe Nyquist, keine Seitenbandwirkung der
zeitvariablen Verstaerkung.

Der Fix ist der Tiefpass in Schritt 0 (0,875x Nyquist, 255 Taps, Kaiser Beta
8) plus ein neu vermessener Detektor: 8x Oversampling, halbe Laenge 16 (32
Taps/Phase), Kaiser Beta 8, Cutoff fest bei 1/Oversampling. Mit dem Tiefpass
gibt es nahe Nyquist praktisch nichts mehr, was der Detektor verpassen
koennte. Gemessene Rest-Ueberschreitung ohne Abstand: echtes Material
<= +0,06 dB, Rauschen <= +0,095 dB, bei 44,1/48/96 kHz und Decken
-0,1/-1,0/-3,0 dBTP.

**Abstand, von Robin festgelegt: 0,2 dB.** Das ist der einzige Abstand, den
diese Datei noch behauptet. Jede andere Kombination aus Oversampling,
Detektor-Halblaenge und Beta hat keine Messung hinter sich und verlangt ein
explizites `safety_db`, sonst wirft `margin_db` einen Fehler statt zu raten.

Latenzkosten: der Tiefpass allein kostet 127 Samples Gruppenlaufzeit
(sample­raten­unabhaengig, weil der Cutoff relativ zu Nyquist definiert ist,
nicht in Hz). Zusammen mit Detektor-Interpolation, halbem Lookahead-Fenster
und halbem Glaettungskernel liegt die Gesamtlatenz bei 48 kHz bei rund 197
Samples, von Robin als ~200 Samples akzeptiert.
