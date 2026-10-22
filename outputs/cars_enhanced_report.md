# Enhancement report - samples/cars.png

## Scene
- severely underexposed image, with flat, low-contrast shadows. mean brightness 6.07, contrast std 22.14, noise estimate 39.0, 97.88% dark pixels, 0.37% blown highlights.

## Attempts
### Round 1 (heuristic-fallback) - did not meet threshold
- plan: gamma=0.45, clahe_clip=2.5, denoise=denoise_nlmeans (strength 9.0), sharpen=0.6
  - gamma=0.45: severely underexposed (mean < 35), strong gamma lift
  - clahe_clip=2.5: somewhat flat contrast (std < 30)
  - denoise_strength=9.0: low measured noise, bumped up since gamma lift will amplify shadow noise
- quality: overall 35.4 (brightness 8.0, contrast 100.0, noise 0.0, clipping 50.5)
- feedback: too dark; noise still visible; shadows crushed

### Round 2 (heuristic-fallback) - did not meet threshold
- plan: gamma=0.35, clahe_clip=2.5, denoise=denoise_nlmeans (strength 13.0), sharpen=0.6
  - gamma=0.45: severely underexposed (mean < 35), strong gamma lift
  - clahe_clip=2.5: somewhat flat contrast (std < 30)
  - denoise_strength=9.0: low measured noise, bumped up since gamma lift will amplify shadow noise
  - reflection adjustments: previous pass still read too dark, dropping gamma further; previous pass still noisy, increasing denoise strength
- quality: overall 43.2 (brightness 28.7, contrast 100.0, noise 0.0, clipping 54.5)
- feedback: too dark; noise still visible; shadows crushed

### Round 3 (heuristic-fallback) - did not meet threshold
- plan: gamma=0.35, clahe_clip=2.5, denoise=denoise_nlmeans (strength 13.0), sharpen=0.6
  - gamma=0.45: severely underexposed (mean < 35), strong gamma lift
  - clahe_clip=2.5: somewhat flat contrast (std < 30)
  - denoise_strength=9.0: low measured noise, bumped up since gamma lift will amplify shadow noise
  - reflection adjustments: previous pass still read too dark, dropping gamma further; previous pass still noisy, increasing denoise strength
- quality: overall 43.2 (brightness 28.7, contrast 100.0, noise 0.0, clipping 54.5)
- feedback: too dark; noise still visible; shadows crushed

### Round 4 (safety-preset) - did not meet threshold
- plan: gamma=0.85, clahe_clip=2.0, denoise=denoise_bilateral (strength 8.0), sharpen=0.3
  - no attempt cleared the quality threshold, applying the conservative safety preset
- quality: overall 37.0 (brightness 0.0, contrast 92.9, noise 29.7, clipping 42.2)
- feedback: too dark; noise still visible; shadows crushed

## Result
- output: outputs/cars_enhanced.jpg
- final score: 37.0
- no attempt cleared the quality threshold, so the conservative safety preset was used for the final output