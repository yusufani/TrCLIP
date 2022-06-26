import matplotlib.pyplot as plt
from PIL import Image
from more_itertools import chunked
from PIL import Image
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import textwrap as twp
from more_itertools import chunked
from PIL import ImageOps, Image
from cairosvg import svg2png
from io import BytesIO
import math
from tqdm import tqdm
import pandas as pd
import os
import io
plt.rc('font', size=8)          # controls default text sizes

def frame(im, thickness=5):
    # Get input image width and height, and calculate output width and height
    iw, ih = im.size
    ow, oh = iw + 2 * thickness, ih + 2 * thickness

    # Draw outer black rounded rect into memory as PNG
    outer = f'<svg width="{ow}" height="{oh}" style="background-color:none"><rect rx="20" ry="20" width="{ow}" height="{oh}" fill="black"/></svg>'
    png = svg2png(bytestring=outer)
    outer = Image.open(BytesIO(png))

    # Draw inner white rounded rect, offset by thickness into memory as PNG
    inner = f'<svg width="{ow}" height="{oh}"><rect x="{thickness}" y="{thickness}" rx="20" ry="20" width="{iw}" height="{ih}" fill="white"/></svg>'
    png = svg2png(bytestring=inner)
    inner = Image.open(BytesIO(png)).convert('L')

    # Expand original canvas with black to match output size
    expanded = ImageOps.expand(im, border=thickness, fill=(0, 0, 0)).convert('RGB')

    # Paste expanded image onto outer black border using inner white rectangle as mask
    outer.paste(expanded, None, inner)
    return outer


def image_retrieval_visualize(per_mode_indices, per_mode_probs, queries, image_paths, n_figure_in_column=2,
                              n_images_in_figure=4, n_figure_in_row=1, save_fig=False, show=True,
                              break_on_index=-1):
    for i, chunk in tqdm(
            enumerate(chunked(zip(per_mode_indices, per_mode_probs, queries), n_figure_in_column * n_figure_in_row)),
            total=int(math.ceil(len(per_mode_probs) / (n_figure_in_column * n_figure_in_row))),
            desc='Generating figures'):
        if break_on_index == i:
            break

        n_row = min(len(chunk), n_figure_in_column)
        n_col = len(chunk) // n_row

        fig = plt.figure(constrained_layout=True, dpi=400, figsize=(n_col * 4, n_row * 2))

        sub_figs = fig.subfigures(n_row, n_col, wspace=0, hspace=0)
        if len(chunk) == 1:
            sub_figs = [sub_figs]

        for row_id, row_fig in enumerate(sub_figs):
            if n_col == 1:
                row_fig = [row_fig]
            for col_id, col_fig in enumerate(row_fig):

                col_fig.patch.set_linewidth(1)
                col_fig.patch.set_edgecolor('#ABA9AC')
                col_fig.patch.set_linestyle('-')

                col_fig.set_alpha(0.01)
                indices, probs, query = chunk[row_id * n_col + col_id]
                col_fig.text(0.5, 0.04,

                             'Number of images processed:' + str(len(probs)),
                             horizontalalignment='center',
                             style='italic',
                             fontsize=7,
                             # weight="bold",
                             bbox={'facecolor': 'coral',
                                   'alpha': 0.8,
                                   'pad': 2,
                                   }, color='black')

                if type(query) == list:
                    wrap_len = 68

                    text1 = twp.fill(f'Text : {query[0]}', wrap_len)
                    text2 = twp.fill(f'(En : {query[1]})', wrap_len)
                    print(text2)
                    query = text1 + '\n' + text2
                else:
                    query = f'Text : {query}'
                query = "\n" + query
                col_fig.suptitle(query)

                axes = col_fig.subplots(1, n_images_in_figure)
                # plt.subplots_adjust(left=0.001 , right=0.99)


                print(f'probs : {probs}')
                print(f'indices : {indices}')

                for ax_id, ax in enumerate(axes):
                    image_path = image_paths[indices[ax_id]]
                    image = Image.open(image_path)
                    image = frame(image, thickness=3)
                    print(f'ax_id : {ax_id}')
                    ax.set_title( "{:.4f}".format(probs[indices[ax_id]]) , fontsize=7)
                    ax.imshow(image)
                    image.close()

                    ax.set_axis_off()

        if save_fig:
            os.makedirs(save_fig, exist_ok=True)
            plt.savefig(save_fig, dpi=300)
        if show:
            plt.show()
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        return Image.open(img_buf)

def text_retrieval_visualize(per_mode_indices, per_mode_probs, queries, texts, n_figure_in_column=2,
                             n_texts_in_figure=4, n_figure_in_row=1, save_fig=True, show=False,
                             break_on_index=-1, auto_trans=False):
    for i, chunk in tqdm(
            enumerate(chunked(zip(per_mode_indices, per_mode_probs, queries), n_figure_in_column * n_figure_in_row)),
            total=int(math.ceil(len(per_mode_probs) / (n_figure_in_column * n_figure_in_row))),
            desc='Generating figures'):
        if break_on_index == i:
            break

        n_row = min(len(chunk), n_figure_in_column)
        n_col = len(chunk) // n_row

        fig = plt.figure(constrained_layout=True, dpi=300, figsize=(n_col * 6, n_row * 2.5))

        sub_figs = fig.subfigures(n_row, n_col, wspace=0, hspace=0)
        if len(chunk) == 1:
            sub_figs = [sub_figs]

        for row_id, row_fig in enumerate(sub_figs):
            if n_col == 1:
                row_fig = [row_fig]
            for col_id, col_fig in enumerate(row_fig):
                col_fig.patch.set_linewidth(1)
                col_fig.patch.set_edgecolor('#ABA9AC')
                col_fig.patch.set_linestyle('-')
                col_fig.set_alpha(0.01)
                indices, probs, query = chunk[row_id * n_col + col_id]

                ax_im, ax_text = col_fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

                ax_im.text(0.5, 1.3,
                           'Number of text\nprocessed:' + str(len(probs)),
                           horizontalalignment='center',
                           verticalalignment='top', transform=ax_im.transAxes, style='italic',
                           fontsize=7,
                           # weight="bold",
                           bbox={'facecolor': 'coral',
                                 'alpha': 0.8,
                                 'pad': 2,
                                 }, color='black')

                image_path = query
                image = Image.open(image_path)
                image = frame(image, thickness=3)
                ax_im.imshow(image )
                image.close()
                ax_im.set_axis_off()
                ax_text.set_axis_off()

                texts_temps = []
                for i in range(n_texts_in_figure):

                    text = texts[indices[i]] + ' Probability: ' + "{:.4f}".format(probs[i])
                    wrap_len = 74
                    text = twp.fill(text, wrap_len)
                    if auto_trans:
                        import translators as ts
                        text_en = f"(En: {ts.google(text, from_language='tr', to_language='en')})"
                        text += '\n' + twp.fill(text_en, wrap_len)
                    texts_temps.append(text)
                df = pd.DataFrame(texts_temps, columns=['Product Title'])

                the_table = ax_text.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left', )
                the_table[(0, 0)].set_facecolor("#40466e")
                the_table[(0, 0)].set_text_props(weight='bold', color='white')

                for i in range(1, n_texts_in_figure + 1):
                    the_table[(i, 0)].set_height(0.3)
                    the_table[(i, 0)].PAD = 0.01

                the_table.auto_set_font_size(False)
                the_table.set_fontsize(8)

        if save_fig:
            os.makedirs(save_fig, exist_ok=True)
            plt.savefig(save_fig, dpi=300)
        if show:
            plt.show()
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        return Image.open(img_buf)
