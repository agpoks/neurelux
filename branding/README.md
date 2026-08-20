# branding/

Logo assets for NeuRelux. The design combines the project's three core physical motifs into one badge: the magnetic flux wave (cyan), the Cauer-ladder skin-effect layers growing shorter with depth (amber → rose bars), and the reluctance-graph edges connecting them (thin lines between the layer nodes) — see `PLAN.md` §3–5 and `notebooks/01_skin_effect_cauer_synthetic.ipynb`.

| File | Use |
|---|---|
| `logo-icon.svg` | Master vector icon. Source of truth — edit this one, then regenerate the others. Embedded in `README.md`; copied to `docs/source/_static/logo.svg` for the Sphinx/ReadTheDocs sidebar (`html_logo` in `docs/source/conf.py`). |
| `logo-icon.png` | 512×512 raster of the icon, for anywhere an SVG isn't accepted. |
| `social-preview.svg` | 1280×640 landscape banner (icon + wordmark + tagline), matching GitHub's recommended social-preview size. |
| `social-preview.png` | Rasterized `social-preview.svg`, ready to upload. |
| `docs/source/_static/favicon.png` | 64×64 raster of the icon, used as `html_favicon` for the docs site. |

## Regenerating the PNGs after editing an SVG

```bash
pip install cairosvg --no-build-isolation   # if not already installed
python3 -c "
import cairosvg
cairosvg.svg2png(url='branding/logo-icon.svg', write_to='branding/logo-icon.png', output_width=512, output_height=512)
cairosvg.svg2png(url='branding/logo-icon.svg', write_to='docs/source/_static/favicon.png', output_width=64, output_height=64)
cairosvg.svg2png(url='branding/social-preview.svg', write_to='branding/social-preview.png', output_width=1280, output_height=640)
"
cp branding/logo-icon.svg docs/source/_static/logo.svg
```

## Setting the GitHub social preview image

GitHub has no API for this — it must be uploaded through the web UI: repository **Settings → General → Social preview → Upload an image**, using `branding/social-preview.png`.
