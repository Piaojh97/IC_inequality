"""Insert the author's original vector icons into the freshly generated Fig. S5.

Run after build_figures.py (requires pypdf). PDF clipping and transformations
preserve vector paths; no artwork is rasterized.
"""
from pathlib import Path
from io import BytesIO
import json
from pypdf import PdfReader, PdfWriter, Transformation

HERE=Path(__file__).resolve().parent


def main():
    target=HERE/"overleaf/ic-model.pdf"
    reader=PdfReader(BytesIO(target.read_bytes()))
    icons=PdfReader(HERE/"source_data/workflow_icons.pdf")
    lookup={"user":0,"item":1,"recommender":2}
    for slot in json.loads((HERE/"results/workflow_icon_positions.json").read_text()):
        source=icons.pages[lookup[slot["name"]]]
        x,y,w,h=slot["rect"]
        sw,sh=float(source.mediabox.width),float(source.mediabox.height)
        scale=min(w/sw,h/sh)
        tx=x+(w-sw*scale)/2
        ty=y+(h-sh*scale)/2
        reader.pages[0].merge_transformed_page(source,
            Transformation().scale(scale).translate(tx,ty))
    writer=PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_metadata({"/Title":"Model dynamics"})
    output=BytesIO()
    writer.write(output)
    target.write_bytes(output.getvalue())
    print("Composed Figure S5 with the author's original vector artwork.")


if __name__=="__main__":
    main()
