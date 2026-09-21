# True Peak (analysis/true_peak.py)

True-Peak- und Inter-Sample-Peak-Messung nach ITU-R BS.1770-4. Messinstrument,
kein Prozessor. Gebaut fuer die TagoClip-Pro-Frage: wie stark schiesst ein
geclipptes Signal zwischen den Samples ueber seinen Sample-Peak hinaus.

## Das Problem

Ein Sample-Peak misst nur die abgetasteten Werte. Der D/A-Wandler rekonstruiert
daraus eine bandbegrenzte kontinuierliche Wellenform, und die kann zwischen zwei
Samples hoeher liegen als jedes einzelne Sample. Dieser Ueberschuss heisst
Inter-Sample-Peak (ISP), das Maximum der rekonstruierten Kurve heisst True Peak,
Einheit dBTP.

Analytischer Worst Case und gleichzeitig der Testfall des Moduls: ein Sinus bei
fs/4, abgetastet mit 45 Grad Phase. Die Samples liegen dann alle bei
`sin(pi/4) = 0.7071`, also -3.01 dBFS, die rekonstruierte Kurve erreicht aber
1.0, also 0 dBTP. Der Ueberschuss betraegt exakt 3.01 dB, ohne dass ein einziges
Sample ueber Full Scale liegt.

## Warum das fuer einen Clipper zaehlt

Clipping erzeugt Obertoene. Ein Teil davon liegt oberhalb Nyquist und faltet
zurueck (Aliasing, dagegen hilft Oversampling oder ADAA). Der Rest bleibt
legitim im Band und macht die Wellenform kantiger. Genau diese Kanten erzeugen
Ueberschwinger in der Rekonstruktion. Ein auf 0 dBFS gefahrener Clipper liefert
also eine Datei, die im Sample-Peak sauber aussieht und trotzdem bei jedem
Lossy-Encoder und jeder Plattform-Pruefung als Ueberschreitung auffaellt.

**Abgrenzung, die das Messdesign traegt:** Aliasing und Inter-Sample-Peaks sind
zwei verschiedene Mechanismen. ADAA unterdrueckt Aliasing, es entfernt aber
nicht die legitimen Obertoene im Band, und damit vermutlich auch nicht den
Ueberschuss. Die Vermutung ist falsifizierbar: senkt ADAA bei gleichem
Oversampling den dBTP im Mittel um mehr als 1.5 dB, ist sie widerlegt. Gemessen
wird das in `examples/isp_sweep.py`, das Ergebnis landet in `docs/measurements/`.

## Verfahren

BS.1770-4 schreibt Oversampling um mindestens Faktor 4 vor, danach Peak des
oversampelten Signals. Hier wird derselbe Ansatz mit dem Resampling-Pfad der
Library gefahren (`scipy.signal.resample_poly`, Kaiser beta 12, etwa 115 dB
Stopband), identisch zu `distortion/clipper.py`, damit Mess- und Prozesskette
dieselbe Interpolation benutzen.

**Default ist 16x, nicht die 4x aus der Norm.** Ein 4x-Meter unterschaetzt den
echten Wert, weil das Maximum zwischen zwei der vier Stuetzstellen liegen kann.
Der Fehler waechst mit der Frequenz und liegt nahe Nyquist in der Groessenordnung
einiger Zehntel dB. Fuer ein Meter im Plugin ist 4x der Kompromiss, als
Schiedsrichter in der Messung ist er zu grob.

Bekannte Grenze: an den Puffergrenzen laeuft der Interpolationsfilter ins Leere,
die ersten und letzten Taps sind dadurch leicht gedaempft. Fuer ganze Dateien
und Testsignale ist das irrelevant, fuer blockweise Messung waere ein Ueberlapp
noetig.

## Status

- Python, offline, ganzer Buffer. Kein Meter fuer den Echtzeitpfad.
- Kein DubCheck-Code. DubCheck misst True Peak ebenfalls nach BS.1770-4, die
  Implementierung hier ist unabhaengig nach oeffentlicher Norm entstanden
  (IP-Regel in CONTRIBUTING.md).
- C++-Promotion noch kein Thema, erst wenn ein Produkt das Meter braucht.
