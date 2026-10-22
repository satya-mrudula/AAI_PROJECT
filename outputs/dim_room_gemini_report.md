# Enhancement report - samples/dim_room.jpg

## Scene
- mildly underexposed image, with visible sensor noise. mean brightness 94.05, contrast std 30.32, noise estimate 2060.9, 12.39% dark pixels, 0.0% blown highlights.

## Attempts
### Round 1 (gemini+rag) - PASSED
- plan: gamma=0.65, clahe_clip=2.5, denoise=denoise_nlmeans (strength 15.0), sharpen=0.5
  - Moderate low-light with high noise requires strong non-local means denoising combined with gamma correction and CLAHE to lift shadows.
- quality: overall 88.8 (brightness 100.0, contrast 92.9, noise 63.1, clipping 98.9)
- feedback: noise still visible

## Result
- output: outputs/dim_room_gemini.jpg
- final score: 88.8