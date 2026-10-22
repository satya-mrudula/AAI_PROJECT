# Enhancement report - samples/dark_alley.jpg

## Scene
- clearly underexposed image, with flat, low-contrast shadows and visible sensor noise. mean brightness 42.98, contrast std 22.24, noise estimate 5763.4, 76.54% dark pixels, 0.0% blown highlights.

## Attempts
### Round 1 (heuristic-fallback) - did not meet threshold
- plan: gamma=0.6, clahe_clip=2.5, denoise=denoise_nlmeans (strength 17.0), sharpen=0.46
  - gamma=0.6: clearly underexposed (mean < 60)
  - clahe_clip=2.5: somewhat flat contrast (std < 30)
  - denoise_strength=17.0: high measured noise, bumped up since gamma lift will amplify shadow noise
- quality: overall 69.8 (brightness 100.0, contrast 92.9, noise 0.0, clipping 77.0)
- feedback: noise still visible

### Round 2 (heuristic-fallback) - PASSED
- plan: gamma=0.6, clahe_clip=2.5, denoise=denoise_nlmeans (strength 21.0), sharpen=0.46
  - gamma=0.6: clearly underexposed (mean < 60)
  - clahe_clip=2.5: somewhat flat contrast (std < 30)
  - denoise_strength=17.0: high measured noise, bumped up since gamma lift will amplify shadow noise
  - reflection adjustments: previous pass still noisy, increasing denoise strength
- quality: overall 72.6 (brightness 100.0, contrast 100.0, noise 0.0, clipping 83.8)
- feedback: noise still visible

## Result
- output: outputs/dark_alley_enhanced.jpg
- final score: 72.6