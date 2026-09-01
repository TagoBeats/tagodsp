# MaskingDetector (analysis/masking.py)

Konflikt-Erkennung zwischen zwei Spuren. Basis fuer X-Ray, Plugin der Tago-Linie
mit Sidechain-Eingang. Portiert aus dem Audiotool-Hackathon-Prototyp
(`project-xray/mockup/analysis/analyze_stems.py`), Clustering 1:1 uebernommen und
gegen den Prototyp verifiziert (26 von 26 Zonen identisch).

## Pipeline

Mono-Summe je Spur, STFT, Leistungsspektrum, Aggregation auf 30 logarithmische
Baender von 20 Hz bis 20 kHz und auf Zeitfenster fester Laenge. Pro Spurpaar
entsteht daraus eine Zellmatrix (Fenster x Band) mit einem Konfliktwert.
Clustering identisch fuer alle Scorings: BFS ueber Zellen ab der Hit-Schwelle,
Bandnachbarschaft 1, Zeitluecken bis 2 Fenster, Mindestdauer 3 Fenster,
Zonenwert als p90 der Region. Bei `relative` und `collision` sind die Zellen
[0, 1] und die Schwellen sind es auch (`hit_min`, `score_min`); bei
`contention` sind die Zellen dB und die Schwellen ebenfalls
(`contention_hit_db`, `contention_score_db`), die Normierung auf [0, 1]
passiert einmal pro ueberlebender Zone.

## Drei Scorings, zwei scheiden aus

- `relative` (der ausgelieferte Prototyp-Detektor): jede Spur auf ihren eigenen
  Peak normiert, `cell = min(norm_a, norm_b)`. **Widerlegt.** Im synthetischen
  Pegel-Sweep (`examples/masking_sweep.py`) liefert er ueber 60 dB
  Pegelunterschied hinweg denselben Score, 0.959 konstant von -30 bis +30 dB. Er
  misst "beide haben hier Energie", nicht Verdeckung.
- `collision`: `cell = 2*min(a,b)/(a+b)` plus Hoerbarkeits-Gate. Behebt die
  Pegelblindheit von `relative`, ist als Verhaeltnis aber massstabsblind:
  75 Prozent seiner Zonen liegen in Baendern 25 bis 35 dB unter der
  Breitbandleistung ihres Fensters, also in unhoerbarem Material, das nur
  punktet weil beide Spuren dort gleich leise sind.
- `contention`: `share = min(a,b) / mix_bb[w]`, als dB-Wert. `min(a,b)` ist bereits die streitende Energie,
  `mix_bb` die Breitbandleistung des Paares im Fenster. Braucht kein
  Hoerbarkeits-Gate, weil `min(a,b)` die Arbeit selbst macht: eine Spur viel
  leiser, beide leise, oder eine allein laut ergeben jeweils einen kleinen Wert.
  Die Obergrenze ist exakt -3.01 dB, weil `min(a,b)` hoechstens die Haelfte von
  `mix_bb` sein kann und diese Schranke mit dem Paar als Referenz nicht nur eine
  obere Grenze, sondern erreichbar ist.

## Referenz ist das Paar, nicht der Mix

`mix_bb` ist die Breitbandleistung der zwei verglichenen Spuren, nicht des
Gesamtmix. Grund: das Produkt ist ein Plugin mit Sidechain-Eingang und sieht nie
mehr als zwei Signale. Gemessener Beleg fuer die Notwendigkeit: mit Mix-Referenz
springt ein Paar von 2 auf 11 Zonen, je nachdem wie viele andere Spuren
mitlaufen. Bewusst in Kauf genommener Preis: zwei leise Spuren, die sich
streiten, punkten auch dann, wenn sie unter dem Gesamtmix untergehen.

## Korpus als Negativkontrolle

51 fertige Beats aus `~/Music`, 5076 Spurpaare (`examples/masking_corpus.py`).
`relative` meldet 8.1 Prozent der Paare, `contention` 4.8 Prozent, auf
90.3 Prozent sind beide still. Ueberlappung der gemeldeten Paare 32.3 Prozent,
die Verfahren sind also nicht redundant, sondern verschiedene Detektoren.

## Summen-Stems, die Falle

Der erste Korpuslauf war kontaminiert. FL-Exporte enthalten den Gesamtmix
zweimal (`<beat>_Master` und `<beat>_Current`, bitidentische Dateien, corr 0.967
gegen die Summe der uebrigen Stems bei 44.5 ms Versatz durch PDC). Ein auf
Stringanfang verankertes Namensmuster greift nie, weil FL jedem Stem den
Projektnamen voranstellt. Betroffen: 49 von 51 Beats, 23.7 Prozent aller Paare.
Wirkung auf das Ergebnis: `contention` sah mit 11.5 Prozent lauter aus als
`relative` mit 11.1, sauber gerechnet ist es mit 4.8 gegen 8.1 das stillere
Verfahren. Die Kontamination hatte die Rangfolge vertauscht.

Erkennung deshalb gemessen statt ueber Namen, siehe `analysis/sum_stems.py`:
Kreuzkorrelation gegen die Summe der uebrigen Stems mit Lag-Suche, plus
Partner-Test gegen gedoppelte Layer. Tests in `tests/test_sum_stems.py`.

## Zwei Hoertests, beide blind, 01.09.2026

Aufbau in beiden: A ist das Zielelement allein, B dasselbe Element plus die
zweite Spur. Kein RMS-Match, nur ein gemeinsamer Clip-Guard-Gain, damit das Ziel
in A und B bitidentisch ist und ausschliesslich der Maskierer dazukommt.

- **Test 1** (`render_listenpack_masking_pairs.py`, Pack
  `2026-08-31_2326_masking_pairs`): gemeldete Paare gegen frequenzgematchte
  stille Kontrollen. Sein Rangurteil ueber `contention` ist **entwertet**, weil
  der Kontrollarm als "meldet keins der Verfahren" definiert war und damit per
  Konstruktion aus `contention`-Fehlschuessen bestand. Methodisch gueltig bleibt:
  gematchte Kontrollen sind Pflicht, sonst trennen sich die Arme nach
  Instrumentengattung, und das Urteil haengt nicht am Pegelabstand.
- **Test 2** (`render_listenpack_mud_vs_burial.py`, Pack
  `2026-08-31_2354_mud_vs_burial`): Matsch gegen Verdeckung, Auswahl
  **detektorblind** rein nach dem Pegelverhaeltnis im Hauptband des Ziels.
  Ergebnis: "geht unter" kam in keiner der zwoelf Positionen vor, auch nicht bei
  17 bis 32 dB Abstand breitbandig. Gleichstand kostet oefter Kontur als
  Verdeckung (3 Gruppen gegen 1, zweimal gleichauf). `contention` gemeldet:
  3 von 4 verschwimmen, still: 2 von 8.

Urteile und Auswertung liegen als `verdicts.md` im jeweiligen Listenpack.

## Entscheidung

`contention` ist die Produktbasis. X-Ray meldet **Matsch** (zwei aehnlich laute
Elemente im selben Band), nicht Verdeckung. Die Blindstelle ab etwa +10 dB
Pegelabstand ist dokumentierte Absicht und kostet gemessen rund ein Drittel der
begrabenen Paare: jede verschwimmende Position, die `contention` verpasst hat,
war ein begrabenes Paar (drei Faelle ueber beide Tests, keine Ausnahme).

## Die Skala, repariert am 01.09.2026

Der Mangel war notiert als "normiert gegen die theoretische Obergrenze statt
gegen einen gemessenen Wert". Die Messung (`examples/masking_scale.py`, 51
Beats) hat das umgedreht: **die Decke ist erreichbar.** Hoechste Zelle im
ganzen Korpus -3.13 dB bei einer exakten Schranke von -3.01. Schuld war der
Boden, `contention_floor_db = -30`, im Code selbst als unvalidierter Platzhalter
markiert. Gemeldet wird nur zwischen -9.75 und -3.10 dB, ein Fenster von
6.65 dB, normiert wurde ueber 27 dB. Die ganze Population sass deshalb im
obersten Viertel: median 0.83, p99 0.96, alles ueber 0.75. Die 2.2 Prozent ueber
0.95 waren genau das p99.

Fix in zwei Teilen. Erstens entscheiden die Schwellen jetzt physikalisch in dB
statt auf einer Anzeigeskala, damit eine Aenderung an der Anzeige die Erkennung
nicht mehr anfassen kann und der C++-Port keine versteckte Abbildung erbt.
Zweitens ist der Anzeigeboden die Meldeschwelle selbst: `Zone.score` normiert
ueber `[contention_score_db, -3 dB]`. Eine Zone genau auf der Schwelle zeigt
0.0, sie ist das Schwaechste, das noch gemeldet wird.

Die Defaults -11.1 dB und -9.75 dB sind die exakten dB-Entsprechungen von
`hit_min` 0.7 und `score_min` 0.75 auf der alten Skala. **Der Zonensatz ist
damit unveraendert der aus Phase 0**, belegt per Werte-Diff und nicht per
gruenem Test:

- `examples/masking_zone_dump.py` vor und nach dem Umbau, 1222 Zonen,
  0 Abweichungen in Spurpaar, Band, Frequenzgrenzen und Fenstergrenzen
- Korpuslauf vor und nach: 5076 Paare, 0 Abweichungen in den Zonenzahlen,
  Summary identisch, `relative` und `collision` in keinem einzigen Wert bewegt
- bewegt haben sich exakt die 242 `contention`-Scores der gemeldeten Paare,
  jeder auf 2e-6 genau auf dem vorhergesagten Umrechnungspfad

Danach: median 0.32, p90 0.66, max 0.985. Eine Schwelle bei 0.5 greift die
obersten 27.7 Prozent der Zonen, bei 0.75 die obersten 4.1 Prozent.

## Die Fragmentierung, geloest am 01.09.2026

Ein gemeldetes Paar kam nicht als ein Befund heraus, sondern als Haufen Zonen:
median 3, maximal 64 (notiert war "median 4, maximal 44"; 44 ist das
zweitschlimmste Paar). Gemessen ueber den Korpus laeuft der Split fast
ausschliesslich entlang der Zeit: 58.8 Prozent der Zonenrelationen innerhalb
eines Paares ueberlappen in der Frequenz und liegen nur zeitlich auseinander,
weitere 38.7 Prozent liegen in beiden Achsen auseinander, eine echte zweite
Frequenzregion zur selben Zeit sind 1.9 Prozent. Der mediane Zeitabstand
zwischen frequenzueberlappenden Fragmenten betraegt 66 Fenster, also 13.2 s.

Damit ist die Ursache benannt: das sind keine zerhackten Nachbarn, die ein
groesseres `gap` verbinden koennte, sondern derselbe Streit, der im Arrangement
wiederkehrt. `gap` hochzudrehen scheidet aus, ein Wert von 66 wuerde jede echte
Pause schlucken und die Zeitausdehnung der Zone bedeutungslos machen.

Loesung ist eine Schicht **ueber** dem Clustering, `summarize_conflicts()`,
gemeldet als `MaskingResult.conflicts`: die Zonen eines Paares werden ueber
ueberlappende Frequenzbereiche gruppiert, die Zeit wird zur Anzahl. 1222 Zonen
werden zu 289 Konflikten, median 1 pro gemeldetem Paar, 83.1 Prozent der Paare
zu genau einem, hoechstens drei. Vorkommen pro Konflikt: median 2, p90 10,
max 44.

Der Konflikt-Score ist das **Maximum** seiner Mitglieder, nicht der Mittelwert.
Er ist der schlimmste Moment des Streits und die einzige Aggregation, die sich
nicht bewegt, wenn dasselbe Material in mehr oder weniger Zonen zerfaellt; ein
Mittelwert sinkt mit der Fragmentzahl und schmuggelt sie so in die Zahl zurueck.
Dauerhaftigkeit bleibt eine eigene Achse (`occurrences`, `active_windows`),
damit eine Anzeige nach Schwere oder nach Betroffenheit sortieren kann, ohne
dass beides vermischt wird.

Der Intervall-Merge ist transitiv und kann Zonen verketten, die einander nicht
ueberlappen. Gemessen passiert das in 33 von 289 Gruppen und weitet den
gemeldeten Bereich um hoechstens 0.66 Oktaven ueber das breiteste Mitglied
hinaus, also zwei Baender, median 0.00.

**Der Zonensatz ist unangetastet**, die Schicht ist rein additiv. Belegt per
Werte-Diff ueber den Korpus, auf den Stem-Ordner gekeyt und nicht auf den
Ordnernamen (drei Beats teilen sich einen Namen): 51 Beats, 1222 Zonen vorher
wie nachher, 0 Abweichungen in Spurpaar, Band, Frequenz- und Fenstergrenzen und
Score.

Messwerte in `docs/measurements/masking_fragmentation_2026-09-01.md`.

## Grenzen der Evidenz

Ein Hoerer, zwoelf Positionen pro Test, keine Wiederholungen, keine Statistik.
Gerichtete Evidenz, kein Beweis. Gehoert wird immer nur das Spurpaar, nie der
Gesamtmix; das entspricht dem Produkt und nicht der Mischsituation.

## C++-Core, 01.09.2026

Portiert nach `cpp/include/tagodsp/masking.hpp`, dazu `fft.hpp` (Radix-2
Cooley-Tukey, damit die Library header-only und abhaengigkeitsfrei bleibt) und
`stft.hpp` (nur Forward). Header-only, JUCE-frei, C++20.

**Nur `contention` ist mitgekommen.** `relative` ist am 31.08. widerlegt worden
und `collision` war der Zwischenkandidat; beide bleiben in der Python-Workbench,
wo Kandidaten hingehoeren. Ein widerlegter Detektor im Produktkern waere nur ein
zweiter Weg, falsch zu liegen.

Verifiziert per Werte-Diff gegen die Python-Seite, nicht per gruenem Test:
`examples/masking_golden.py` schreibt `cpp/tests/golden/masking.txt`,
`cpp/tests/test_masking.cpp` baut dieselbe Fixture aus der Spec neu und
vergleicht jede Stufe. Gemessene groesste Abweichung:

| Stufe | Abweichung |
| --- | --- |
| STFT-Rohzeilen | 6.3e-16 vom Frame-Peak |
| Band-Power-Gitter | 4.4e-16 vom Gitter-Peak |
| `contention`-Zellen | 3.0e-11 dB absolut |
| Zonen und Konflikte | identisch in Band, Fenster, Frequenzgrenzen, Score |

Der Vergleich laeuft gegen eine Toleranz und nicht auf Bit-Gleichheit, weil
beide Seiten `sin()` durch verschiedene Bibliotheken rechnen. Die Bezugsgroesse
ist bewusst der lauteste Wert des Laufs und nicht der Wert der einzelnen Zelle:
in einer spektralen Null steht der Rest einer fast vollstaendigen Ausloeschung,
ein relativer Vergleich misst dort Rundungsrauschen statt Uebereinstimmung.
Zonen- und Konfliktzahlen werden zusaetzlich exakt geprueft, denn eine Toleranz
kann einen Wertedrift verdecken, aber keine gekippte Schwellenentscheidung.

Die Fixture ist synthetisch und zwei Sekunden lang: zwei Spuren, die sich
zeitlich zweimal ueberlappen und in zwei getrennten Frequenzregionen (um 260 Hz
und um 980 Hz) streiten, plus eine Rauschspur, die schweigen muss. Ergebnis 4
Zonen, 2 Konflikte, jeder aus zwei zeitlich getrennten Zonen. Damit laeuft der
Frequenz-Merge der Zusammenfassungsschicht wirklich durch und wird nicht nur
angenommen.

Beide Konfliktregionen liegen mit Absicht deutlich ueber dem tiefen Ende. Bei
`n_fft = 1024` und `sr = 16000` ist ein Bin 15.6 Hz breit, ein 70-Hz-Ton
verschmiert seine Hauptkeule ueber drei log-verteilte Baender und keine der
beiden Spuren streitet dort noch sauber mit der anderen. Das ist eine
Eigenschaft der Fixture, nicht des Detektors, haette das Golden aber von
Leakage statt vom getesteten Code abhaengig gemacht.

## Status

- Python-Prototyp, offline (`analyze()` auf ganzen Buffern, kein Block-State);
  der C++-Core ist derselbe Offline-Schnitt. Der Umbau auf Block-State und
  Hintergrund-Thread gehoert in den Plugin-Schritt, nicht in den Port.
- Die Schwellen stammen der Hoehe nach immer noch aus `relative` (0.7 / 0.75,
  umgerechnet in -11.1 dB / -9.75 dB) und sind fuer `contention` als *Hoehe*
  unvalidiert. Was validiert ist, ist der Zonensatz, den sie erzeugen: genau
  den hat Hoertest 2 beurteilt.
- Skripte: `examples/masking_sweep.py` (Positivkontrolle),
  `examples/masking_corpus.py` (Negativkontrolle in Serie),
  `examples/masking_scale.py` (Verteilung der Skala),
  `examples/masking_zone_dump.py` (Werte-Diff ueber alle Zonenfelder), die
  beiden Listenpack-Renderer fuer die Hoertests
- Keine offenen Anzeige-Maengel mehr; der C++-Port ist gezogen, siehe oben
