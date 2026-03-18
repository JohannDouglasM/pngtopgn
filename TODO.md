# TODO

1. If possible, reverse engineer the corner locations for the new 500 images of the green board (chess-dataset) and add them to our dataset.
2. Test if we can drop the square detection input channel (ch2 heatmap) — if performance holds, inference goes from ~30s to <1s.
3. Try heatmap output with peak-finding/argmax instead of direct coordinate regression for corner detection.
