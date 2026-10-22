# Enhancement report - samples/dim_room.jpg

## Scene
- mildly underexposed image, with visible sensor noise. mean brightness 94.05, contrast std 30.32, noise estimate 2060.9, 12.39% dark pixels, 0.0% blown highlights.

## Attempts
### Round 1 (heuristic-fallback) - PASSED
- plan: gamma=0.88, clahe_clip=2.0, denoise=denoise_nlmeans (strength 14.0), sharpen=0.52
  - gamma=0.88: mildly dark (mean < 95), light gamma touch
  - clahe_clip=2.0: contrast in a reasonable range, default clip
  - denoise_strength=14.0: high measured noise
- quality: overall 89.3 (brightness 100.0, contrast 92.8, noise 65.7, clipping 98.0)
- feedback: noise still visible

## Result
- output: outputs/dim_room_enhanced.jpg
- final score: 89.3