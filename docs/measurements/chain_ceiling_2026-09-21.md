# Haelt die Kette ihre Decke?

Erzeugt von `examples/chain_ceiling_check.py`, sr 44100 Hz, 1 s pro Signal, 324 Kombinationen. Ziel ist -1 dBTP, gegengeprueft mit 32x nach BS.1770-4 und damit feiner als der Detektor des Limiters selbst, sonst teilte die Pruefung dessen blinden Fleck.

Die erste Messung der Prototyp-Phase hatte gezeigt, dass der True Peak nach dem Clipping bis zu 5,8 dB ueber der Decke liegt und Oversampling daran nichts aendert. Hier dieselbe Frage einmal ohne und einmal mit der Limiter-Stufe am Ende.

## Ueberschreitungen der Decke

| Clipper-Stufe | ohne Limiter | mit Limiter | schlimmster Fall ohne | mit |
| :--- | :--- | :--- | :--- | :--- |
| plain 1x | **90 von 108** | **0 von 108** | +5.77 dBTP | -1.03 dBTP |
| plain 8x | **85 von 108** | **0 von 108** | +4.17 dBTP | -1.03 dBTP |
| ADAA 4x | **85 von 108** | **0 von 108** | +4.18 dBTP | -1.03 dBTP |

## Der schlimmste Fall mit Limiter

-1.029 dBTP bei noise, Kurve fl, Threshold 0.50, Drive +6 dB, Stufe plain 1x.

Der Plan hatte 0,1 dB Schlupf ueber der Decke erlaubt, also -0.9 dBTP. Gefordert wird hier die schaerfere Fassung: kein Wert ueber der gesetzten Zahl. Moeglich macht das der Sicherheitsabstand von 0.05 dB, der dem Detektor mit 16x folgt, siehe `docs/concepts/true_peak_limiter.md`.
