# MonoLow (stereo/mono_low.py)

Stereo-Utility fuer TagoClip: alles unterhalb einer Grenzfrequenz mono summieren
(808-/Mixbus-Klassiker gegen phasiges Low-End auf Clubanlagen und Vinyl).

## Ansatz

Linkwitz-Riley 4. Ordnung pro Kanal (zwei kaskadierte RBJ-Butterworth-Biquads
je Band), Low-Band beider Kanaele gemittelt, High-Band bleibt stereo:

    out_i = (LR4_lp(L) + LR4_lp(R)) / 2 + LR4_hp(x_i)

LR4 summiert allpass-flach und in Phase: Mono-Material passiert mit flacher
Magnitude (Test: < 0.2 dB ueber 30 Hz - 18 kHz), nur die Allpass-Phase des
Crossovers bleibt. Quellen: Linkwitz JAES 1976, RBJ Audio EQ Cookbook.

## Erwartbares Verhalten (nicht "Totalausloeschung")

Side-Anteile unterhalb fc lecken durch die Highpass-Flanke (24 dB/Okt):
bei fc=120 Hz sind es ~30 dB Reduktion bei 50 Hz, ~45 dB bei 30 Hz.
Das ist Standard bei Crossover-basierten Mono-Makern; steilere Varianten
(LR8, linear phase) nur bauen wenn Robins Ohr das LR4 ablehnt.

## Status

- Python-Prototyp, offline; Listenpack mit detuned-808 und Pad+808-Mix
- C++-Promotion zusammen mit dem Clipper im TagoClip-Plugin-Kontext
