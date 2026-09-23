# Wavefolder (distortion/folder.py)

Kandidat fuer eine vierte Kurve neben `fl`, `hard` und `tanh`. Vorschlag von
Raphael Kim (Mail 22.09.2026). Nicht als eigener Modus, sondern als weiterer
Eintrag in derselben Kennlinien-Registry, damit Signalweg, ADAA-Maschinerie und
Oversampling unveraendert bleiben.

**Stand 23.09.2026: fuer TagoClip Pro abgelehnt, nicht an der Messung.** Der
Positionierungssatz dieses Produkts ist "The FL clipper, everywhere", und ein
Folder ist ein anderes Verzerrungsprinzip. Der Code bleibt hier liegen und der
Befund ist an das geparkte Waveshaper-Produkt uebergegangen, wo eine frei
formbare Kennlinie der Produktkern ist. Was unten steht, gilt unveraendert
weiter und ist die Vorarbeit fuer diesen Fall.

## Was eine Faltung von einer Begrenzung unterscheidet

Ein Clipper drueckt das Signal an der Schwelle flach, ein Folder spiegelt es
zurueck. Bei hohem Drive passiert das mehrfach, das Ausgangssignal laeuft durch
null und kommt mit umgekehrtem Vorzeichen wieder heraus. Die Obertoene wachsen
deshalb nicht monoton mit dem Pegel, sie kommen und gehen mit der Anzahl der
Faltungen, durch die das momentane Sample laeuft. Das ist der metallische,
vokalartige West-Coast-Klang (Buchla, Serge).

## Zwei Formen, absichtlich beide

Gleiche Faltgeometrie, einziger Unterschied ist die Glattheit:

    fold      y = sign(x) * t * T(|x| / t)     Dreieck, Ecke an jedem Faltpunkt
    sinefold  y = t * sin(x / t)               C-unendlich, keine einzige Ecke

`T` ist die Dreieckschwingung der Periode 4. Beide haben Steigung 1 im
Ursprung, beide erreichen ihren Scheitel genau beim Threshold, wie `hard` in
dieser Familie und anders als `fl` und `tanh` mit ihrer festen Decke bei 1,0.
Unterhalb der Schwelle ist die Dreiecksform exakt die Identitaet, die
Sinusform ein weiches Knie: `sin(1) = 0,841`, also 1,5 dB Kompression direkt
vor der ersten Faltung.

Die beiden Formen existieren nebeneinander, weil der Gate sonst zwei Fragen auf
einmal beantwortet haette: ob Wavefolding aliast, oder ob Ecken aliasen.

**Verworfen wurde eine dritte Parametrisierung**, die nur im Kopfraum oberhalb
von `t` faltet und die Decke bei 1,0 laesst. Beim Familien-Default 100/128
pendelt der Ausgang dort zwischen 0,56 und 1,0. Das ist eine Textur auf einem
Clipper und kein Folder. Faltungstiefe muss der ganze Bereich sein.

## Skew

Symmetrisch gefaltet ist die Kennlinie ungerade und erzeugt nur ungeradzahlige
Obertoene. Skew macht die beiden Haelften unterschiedlich und bringt geradzahlige
dazu. Die Asymmetrie sitzt im Threshold, nicht in der Faltbreite:

    t_pos = t,  t_neg = t * (1 - skew),  skew in [0, 1)

Negativer Skew ist ausgeschlossen: das ist dieselbe Kurve mit gedrehter
Polaritaet und wuerde den negativen Scheitel ueber den Threshold schieben.

## Stammfunktion

Beide Formen haben ein geschlossenes `F1` ohne Fallunterscheidung im Nenner:

    fold:      F1 = t^2 * G(|x| / t)        G = Integral der Dreieckschwingung
    sinefold:  F1 = t^2 * (1 - cos(x / t))

`F1` ist nur bei Skew 0 gerade, wird also auf vorzeichenbehaftetem `x`
ausgewertet und waehlt seine Seite selbst. Das ist der einzige Punkt, an dem
sich die Kandidaten von den drei Clipping-Kurven in `adaa.py` unterscheiden.

## Gate-Ergebnis (23.09.2026)

Volle Messung in `docs/measurements/fold_gate_2026-09-23.md`. Kurz:

- **Kriterium 1a, Alias-Niveau bei ADAA 4x.** Grenze war `hard` plus 6 dB Luft,
  im selben Lauf gemessen: -39,97 dB. Bestanden nur von `sinefold` (-50,34).
  `fold` -25,28, `skewfold` -23,35, `skewsinefold` -38,83.
- **Kriterium 1b, greift ADAA ueberhaupt.** Alle vier bestanden, Gewinn von
  ADAA 4x gegen plain 1x liegt bei 25,4 bis 34,5 dB gegen die geforderten 15.
  Die Implementierung ist also nicht das Problem, die Kurvennatur ist es.
- **Kriterium 2, eps-Sweep an den Faltpunkten.** Bestanden, aber mit einem
  Unterschied zwischen den Formen, der genau die Theorie bestaetigt: der
  Sprung zwischen Quotient und Mittelpunkt-Zweig faellt bei den Dreiecksformen
  mit 20 dB je Dekade eps, bei den glatten mit 40. An einer Ecke ist der Fehler
  des Mittelpunkt-Zweigs erster Ordnung in eps, sonst zweiter. Beim
  ausgelieferten eps 1e-4 heisst das -92,0 dBFS fuer `fold` gegen die Grenze
  -90, und -194,6 dBFS fuer `sinefold`.
  **Konsequenz, falls eine Dreiecksform je ins Produkt geht:** eps muss dort
  kleiner sein. Bei 1e-6 liegt der Sprung bei -132 dBFS und die Ausloeschung
  immer noch bei -211, es gibt also 40 dB geschenkt und nichts zu bezahlen.
  Die Spreizung des Alias-Werts ueber eps ist bei allen vier 0,00 dB.
- **Nicht im Kriterium, aber im Report:** die schlechteste Einzelzelle. Die
  bestandene Kurve steht bei +24 dB Drive auf -17,07 dB, waehrend die drei
  Clipping-Kurven dort bei -34 dB liegen. Der Mittelwert besteht, der obere
  Rand des Drive-Bereichs nicht.

## Zwei Befunde am Messinstrument

**Erstens, der Nenner.** `alias_nmr_db` misst gegen den Grundton. Ein Folder
loescht seinen eigenen Grundton aus: dessen Amplitude laeuft mit `J1(A/t)`, und bei Threshold 0,5234
und +6 dB liegt `A/t = 3,81` auf der ersten Nullstelle. Der Grundton faellt um
34 dB, der Ausgang bleibt gleich laut, und eine saubere Kurve meldet sich als
dreckig. Deshalb gibt es `alias_to_harmonics_db`, das gegen die gesamte
harmonische Reihe misst. Bei den Clipping-Kurven stimmen beide Metriken auf ein
dB ueberein, der Vergleich mit aelteren Reports bleibt also gueltig.

**Zweitens, die Referenz.** Der eps-Sweep misst eine Differenz zweier fast
gleicher Zahlen, braucht also eine Referenz, die feiner ist als das Gemessene.
Der erste Anlauf nahm `np.longdouble`, und auf Apple Silicon ist das schlicht
`float64`: dieselbe Genauigkeit mit anderem Namen. Die Ausloeschungs-Spalte war
damit konstant null und hat nichts gemessen. Jetzt rechnet die Referenz exakt,
per `Fraction` fuer die stueckweise quadratische Dreiecksform und per
Summen-zu-Produkt-Identitaet fuer die Sinusform, die dabei ohne Subtraktion
auskommt.

## Offen

- Hoerabnahme nach Raphaels Rezept, `examples/render_listenpack_fold.py`.
  Entscheidet, ob die glatte Form ueberhaupt nach Folder klingt und ob Skew
  seine 11 dB Alias wert ist.
- Gleichanteil bei den schiefen Formen: `skewfold` liegt bei -31,6 dBFS. Falls
  eine schiefe Kurve ins Produkt geht, braucht sie einen DC-Blocker.
