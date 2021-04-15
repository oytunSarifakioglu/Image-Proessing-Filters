import tkinter as tk
import numpy as np
import math
from tkinter import *
from tkinter import filedialog
import cv2
from skimage import data, io, filters, feature, color, segmentation, util,img_as_float
from skimage.future import graph
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt 
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, disk
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.filters import unsharp_mask
from skimage.segmentation import watershed
from skimage.transform import warp, swirl, PiecewiseAffineTransform, rotate,rescale
from skimage.exposure import rescale_intensity
from skimage.morphology import max_tree_local_maxima
from skimage.filters import rank
from scipy import ndimage as ndi
from skimage import transform as tf
from skimage.feature import CENSURE
from matplotlib import pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from skimage.exposure import match_histograms
from skimage.segmentation import (morphological_chan_vese, morphological_geodesic_active_contour,inverse_gaussian_gradient,checkerboard_level_set)
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap

    
class Root(Tk):

    def __init__(self):
        super(Root, self).__init__()
        self.title("GÖRÜNTÜ İŞLEME")
        self.minsize(500, 700)
        self.img = data.coins()
        self.img_gray = data.coins()
        self.retina = data.retina()
        self.moon = data.moon()
        self.horse = data.horse()
        self.camera = data.camera()
        self.width = 50
        self.imageGlobal = data.retina()
        self.imageGlobal = cv2.cvtColor(self.imageGlobal, cv2.COLOR_BGR2GRAY)

        self.labelFrame = tk.LabelFrame(self, text="DOSYA AÇ")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)

        self.filterFrame = tk.LabelFrame(self, text="FİLTRELER")
        self.filterFrame.grid(column=0, row=2, padx=20, pady=20)

        self.histogramFrame = tk.LabelFrame(self, text="HİSTOGRAM EŞİTLEME")
        self.histogramFrame.grid(column=0, row=3, padx=20, pady=20)

        self.transformFrame = tk.LabelFrame(self, text="DÖNÜŞTÜRME İŞLEMLERİ")
        self.transformFrame.grid(column=0, row=4, padx=20, pady=20)

        self.videoFrame = tk.LabelFrame(self, text="VİDEO FİLTRESİ")
        self.videoFrame.grid(column=0, row=5, padx=20, pady=20)

        self.intensityFrame = tk.LabelFrame(self, text="YOĞUNLUK İŞLEMLERİ")
        self.intensityFrame.grid(column=1, row=4, padx=20, pady=20)

        self.morphologyFrame = tk.LabelFrame(self, text="MORFOLOJİK İŞLEMLER")
        self.morphologyFrame.grid(column=1, row=3, padx=20, pady=20)
        
        self.activeFrame = tk.LabelFrame(self, text="ACTİVE CONTOUR")
        self.activeFrame.grid(column=1, row=1, padx=20, pady=20)
        
        self.socialFrame = tk.LabelFrame(self, text="SOSYAL MEDYA EFEKTİ")
        self.socialFrame.grid(column=1, row=2, padx=20, pady=20)

        self.upload_image_button()
        self.filterButton()
        self.histogramButton()
        self.transformButton()
        self.videoButton()
        self.intensityButton()
        self.morphologyButton()
        self.activeContourButton()
        self.socialButton()

    def morphologyButton(self):
        self.morphology_button = tk.Button(self.morphologyFrame, text="Morfoloji Ekranını Aç", width=self.width,
                                           command=self.open_morphology_window)
        self.morphology_button.grid(column=0, row=3)

    def intensityButton(self):
        self.intensity_button = tk.Button(self.intensityFrame, text="Yoğunluk İşlemini Aç", width=self.width,
                                          command=self.open_intensity_window)
        self.intensity_button.grid(column=2, row=2)

    def videoButton(self):
        self.video_button = tk.Button(self.videoFrame, text="Kamerayı Aç ('esc' ile çıkış) ", width=self.width,
                                      command=lambda: video())
        self.video_button.grid(column=1, row=2)

    def transformButton(self):
        self.transform_button = tk.Button(self.transformFrame, text="Dönüştürme İşlemini Aç",
                                          width=self.width,
                                          command=self.open_transform_operations)
        self.transform_button.grid(column=0, row=2)

    def histogramButton(self):
        self.filter_button = tk.Button(self.histogramFrame, text='Histogram Eşitleme Örneği', width=self.width,
                                       command=lambda: histogram_matching(self.img))
        self.filter_button.grid(column=2, row=2)

    def filterButton(self):
        self.filter_button = tk.Button(self.filterFrame, text='Filtreleri Aç', width=self.width,
                                       command=self.open_filter_window)
        self.filter_button.grid(column=1, row=1)
        
    def upload_image_button(self):
        self.button = tk.Button(self.labelFrame, text="Dosya Aç", width=self.width, command=self.fileDialog)
        self.button.grid(column=1, row=1)
        
    def activeContourButton(self):
        self.button=tk.Button(self.activeFrame,text='Active Contour Örneği',width=self.width,
                                             command=lambda:active_contour(self.img))
        self.button.grid(column=2,row=2)
        
    def socialButton(self):
        self.social_media_button=tk.Button(self.socialFrame,text='Sosyal Medya Filtresi Örneği',width=self.width,
                                             command=lambda:social_media(self.img))
        self.social_media_button.grid(column=2,row=2)
    
        
    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Dosya Seç", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.label = tk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        if not (self.filename is NONE):
            self.label.configure(text=self.filename)
            self.img = cv2.imread(self.filename)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img_gray = cv2.imread(self.filename, 0)
            self.globalImage = cv2.imread(self.filename)
            self.globalImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.gloabalImg_gray = cv2.imread(self.filename, 0)
             

    def open_transform_operations(self):
        transform_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        transform_window.title('Transform Operations')
        transform_window.geometry("400x400")

        swirl_button = tk.Button(transform_window, text='Girdap İşlemi', width=self.width,
                                 command=lambda: swirled(self.img))
        swirl_button.pack()

        swirl_with_checker_board_button = tk.Button(transform_window, text='Checker Board ile Girdap İşlemi',
                                                    width=self.width,
                                                    command=lambda:  warp_censure(self.img))
        swirl_with_checker_board_button.pack()

        rescale_button = tk.Button(transform_window, text="Kenar Yumuşatma ile Yeniden Ölçeklendirme", width=self.width,
                                   command=lambda: warp_basics(self.img))
        rescale_button.pack()

        resize_button = tk.Button(transform_window, text="Kenar Yumuşatma ile İşlemi Yeniden Boyutlandırma", width=self.width,
                                  command=lambda: forward_iradon(self.img))
        resize_button.pack()

        downscale_button = tk.Button(transform_window, text="Ölçek Küçültme İşlemi", width=self.width,
                                     command=lambda: warp_piecewise_affine(self.img))
        downscale_button.pack()


    def open_intensity_window(self):
        intensity_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        intensity_window.title('YOĞUNLUK İŞLEMİ')
        width = 50
        #tk.Label(intensity_window, width=int(width / 2), text="Değer Girin").grid(row=0)

        label_value = tk.Entry(intensity_window)
        label_value.grid(row=0, column=1)
        out_range = label_value.get()

        if (out_range is not float and out_range is not int):
            out_range = 0.4

        wrapping_button = tk.Button(intensity_window, text="Paketleme İşlemi", width=int(width / 2),
                                    command=lambda: intense(out_range))
        wrapping_button.grid(row=1)

    def open_morphology_window(self):
        morphology_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        morphology_window.title('MORFOLOJİK İŞLEMLER')
        morphology_window.geometry("400x400")
        width = 50

        white_tophat_erosion_button = tk.Button(morphology_window, text="Erosion İşlemi", width=width,
                                   command=lambda:white_tophat_erosion(self.moon))
        white_tophat_erosion_button.pack()

        white_tophat_dilation_button = tk.Button(morphology_window, text="Dilation İşlemi", width=width,
                                    command=lambda: white_tophat_dilation(self.moon))
        white_tophat_dilation_button.pack()

        white_tophat_button = tk.Button(morphology_window, text="White Tophot İşlemi", width=width,
                                   command=lambda: white_tophat_(self.moon))
        white_tophat_button.pack()

        black_tophat_button = tk.Button(morphology_window, text="Black Tophat İlemi", width=width,
                                   command=lambda:black_tophat_(self.moon))
        black_tophat_button.pack()

        skeletonize_button = tk.Button(morphology_window, text="Skeletonize İşlemi", width=width,
                                        command=lambda:skeletonize_(self.moon))
        skeletonize_button.pack()

        convex_hull_button = tk.Button(morphology_window, text="Convex Hull İşlemi", width=width,
                                        command=lambda: convex_hull(self.moon))
        convex_hull_button.pack()

        local_gradient_button = tk.Button(morphology_window, text="Local Gradient İşlemi", width=width,
                                       command=lambda: local_gradient(self.horse))
        local_gradient_button.pack()

        markers_button = tk.Button(morphology_window, text="Markers İşlemi", width=width,
                                             command=lambda:markers(self.horse))
        markers_button.pack()

        segmented_button = tk.Button(morphology_window, text="Segmented İşlemi", width=width,
                                     command=lambda:segmented(self.camera))
        segmented_button.pack()

        maksimum_tree_button = tk.Button(morphology_window, text="Maximum Tree İşlemi", width=width,
                                   command=lambda:maksimum_tree(self.camera))
        maksimum_tree_button.pack()

    def open_filter_window(self):

        filter_window = tk.Tk(screenName=None, baseName=None, className='Tk', useTk=1)
        filter_window.title('FİLTRELER')
        filter_window.geometry("500x500")
        width = 35

        sobel_h_button = tk.Button(filter_window, text='Sobel Filtresi', width=width,
                        command = lambda:sobel(self.img_gray))
        sobel_h_button.pack()

        median_button = tk.Button(filter_window, text='Median Filtresi', width=width,
                                          command=lambda: median(self.img_gray))
        median_button.pack()

        unsharp_masking_button = tk.Button(filter_window, text='Unsharp Masking Filtresi' , width=width,
                                             command=lambda: unsharp_masking(self.img_gray))
        unsharp_masking_button.pack()

        region_boundry_button = tk.Button(filter_window, text='Region Boundary Filtresi', width=width,
                                   command=lambda: region_boundry(self.img_gray))
        region_boundry_button.pack()

        try_all_threshold_filter_button = tk.Button(filter_window,
                                           text='Tüm Tresholdlar',
                                           width=width, command=lambda: try_all_threshold_filter(self.img_gray))
        try_all_threshold_filter_button.pack()

        roberts_filter_button = tk.Button(filter_window, text='Roberts Filtresi', width=width,
                                                command=lambda: roberts_filter(self.img_gray))
        roberts_filter_button.pack()

        prewitt_v_button = tk.Button(filter_window, text='Prewitt Filtresi', width=width,
                                         command=lambda:prewitt_v(self.img_gray))
        prewitt_v_button.pack()

        hysteresis_threshold_button = tk.Button(filter_window, text='Hysteresis Treshold Filtresi', width=width,
                                         command=lambda:  hysteresis_threshold(self.img_gray))
        hysteresis_threshold_button.pack()

        scharr_button = tk.Button(filter_window, text='Scharr Filtresi', width=width,
                                                 command=lambda: scharr_filter(self.img_gray))
        scharr_button.pack()
        
        find_regular_segments_button = tk.Button(filter_window, text='Düzenli Segment Belirleme', width=width,
                                                 command=lambda: find_regular_segments(self.img_gray))
        find_regular_segments_button.pack()

    

#1. Filtre
def sobel(image, mask=None):
    edge_roberts = filters.roberts(image)
    edge_sobel = filters.sobel(image)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
    ax[1].set_title('Sobel Edge Detection')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    edges = filters.sobel(image)



    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')

    ax[1].imshow(edges)
    ax[1].set_title('Sobel edges')



    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()

    plt.show()

#2. Filtre
def median(image):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    med = filters.median(image,disk(5))


    ax[0].imshow(image, cmap = 'gray')
    ax[0].set_title('Original image')

    ax[1].imshow(med, cmap='gray')
    ax[1].set_title('Median Image')

    plt.tight_layout()

    plt.show()

#3. Filtre
def unsharp_masking(image):
    #image = imageGlobal
    result_1 = unsharp_mask(image, radius=1, amount=1)
    result_2 = unsharp_mask(image, radius=5, amount=2)
    result_3 = unsharp_mask(image, radius=20, amount=1)

    fig, axes = plt.subplots(nrows=1, ncols=2,
                             sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original image')

    ax[1].imshow(result_3, cmap=plt.cm.gray)
    ax[1].set_title('Enhanced image, radius=20, amount=1.0')

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()

#4. Filtre
def region_boundry(image):
    
    #image = imageGlobal

    labels = segmentation.slic(image, compactness=30, n_segments=400)
    edges = filters.sobel(image)
    edges_rgb = color.gray2rgb(edges)

    g = graph.rag_boundary(labels, edges)
    lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                        edge_width=1.2)

    plt.colorbar(lc, fraction=0.03)
    io.show()


#5. Filtre
def try_all_threshold_filter(image):
    new_img = filters.try_all_threshold(image, figsize=(15,30 ), verbose=True)
    plt.show()

#6.Filtre
def roberts_filter(image):
    corners = filters.roberts(image)

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(corners)
    ax[1].set_title("Image with Roberts filter")

    plt.tight_layout()
    plt.show()

#7.Filtre 
def prewitt_v(image):
    #image = imageGlobal
    edge_roberts = filters.roberts(image)
    edge_sobel = filters.sobel(image)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                           figsize=(8, 4))

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
    ax[1].set_title('Sobel Edge Detection')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()
    
#8. Filtre
def hysteresis_threshold(image):
    fig, ax = plt.subplots(nrows=3)
    sobel_img = filters.sobel(image)
    new_img = filters.apply_hysteresis_threshold(sobel_img, 0.1, 0.35)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(sobel_img)
    ax[1].set_title('Sobel image')

    ax[2].imshow(new_img, cmap='magma')
    ax[2].set_title('Hyteresis Threshold')

    plt.tight_layout()
    plt.show()
    
#9. Filtre
def scharr_filter(image):
    new_img = filters.scharr(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original image")

    ax[1].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Image with Scharr filter")

    plt.tight_layout()
    plt.show()

#10. Filtre
def find_regular_segments(image):
    original_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    edges = filters.sobel(image)

    grid = util.regular_grid(image.shape, n_points=468)

    seeds = np.zeros(image.shape, dtype=int)
    seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

    w0 = watershed(edges, seeds)
    w1 = watershed(edges, seeds, compactness=0.01)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    ax0.imshow(original_img)
    ax0.set_title("Original image")

    ax1.imshow(color.label2rgb(w0, image))
    ax1.set_title('Classical watershed')

    ax2.imshow(color.label2rgb(w1, image))
    ax2.set_title('Compact watershed')

    plt.show()
    
    
#Sosyal Medya Filtresi
def social_media(image):

    
    grayscale_image = img_as_float(data.camera()[::2, ::2])
    image = color.gray2rgb(grayscale_image)
    
    red_multiplier = [1, 0, 0]
    yellow_multiplier = [1, 1, 0]
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4),
                                   sharex=True, sharey=True)
    ax1.imshow(red_multiplier * image)
    ax2.imshow(yellow_multiplier * image)

    def colorize(image, hue, saturation=1):
        hsv = color.rgb2hsv(image)
        hsv[:, :, 1] = saturation
        hsv[:, :, 0] = hue
        return color.hsv2rgb(hsv)
    hue_rotations = np.linspace(0, 1, 6)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    
    for ax, hue in zip(axes.flat, hue_rotations):
        # Turn down the saturation to give it that vintage look.
        tinted_image = colorize(image, hue, saturation=0.3)
        ax.imshow(tinted_image, vmin=0, vmax=1)
        ax.set_axis_off()
    fig.tight_layout()
    from skimage.filters import rank
    
    top_left = (slice(25),) * 2
    bottom_right = (slice(-25, None),) * 2
    
    sliced_image = image.copy()
    sliced_image[top_left] = colorize(image[top_left], 0.82, saturation=0.5)
    sliced_image[bottom_right] = colorize(image[bottom_right], 0.5, saturation=0.5)
    
    noisy = rank.entropy(grayscale_image, np.ones((9, 9)))
    textured_regions = noisy > 4.25
    masked_image = image.copy()
    masked_image[textured_regions, :] *= red_multiplier
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                   sharex=True, sharey=True)
    ax1.imshow(sliced_image)
    ax2.imshow(masked_image)
    
    plt.show()
    
    
#Histogram Eşitleme
def histogram_matching(image):
    img = image
    reference = data.coffee()

    matched = match_histograms(img, reference, multichannel=True)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(img)
    ax1.set_title('Source')
    ax2.imshow(reference)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show()

def adjust_log(image,nbins=256):
    def plot_img_and_hist(image, axes, bins=256):
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    img = image

    gamma_corrected = exposure.adjust_gamma(img, 2)

    logarithmic_corrected = exposure.adjust_log(img, 1)

    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 3), dtype=np.object)
    axes[0, 0] = plt.subplot(2, 3, 1)
    axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = plt.subplot(2, 3, 4)
    axes[1, 1] = plt.subplot(2, 3, 5)
    axes[1, 2] = plt.subplot(2, 3, 6)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
    ax_img.set_title('Gamma correction')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])
    ax_img.set_title('Logarithmic correction')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    fig.tight_layout()
    plt.show()

#Yoğunluk İşlemleri
def intense(image, in_range='image', out_range='dtype'):
    
    cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
    cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                 'saddlebrown'])
    cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                                   'white'])
    
    ihc_rgb = data.immunohistochemistry()
    ihc_hed = rgb2hed(ihc_rgb)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(ihc_rgb)
    ax[0].set_title("Original image")
    
    ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)
    ax[1].set_title("Hematoxylin")
    
    ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
    ax[2].set_title("Eosin")
    
    ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
    ax[3].set_title("DAB")
    
    for a in ax.ravel():
        a.axis('off')
    
    
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    zdh = np.dstack((np.zeros_like(h), d, h))
    
    fig = plt.figure()
    axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
    axis.imshow(zdh)
    axis.set_title("Stain separated image (rescaled)")
    axis.axis('off')
    plt.show()

    
#Uzay Dönüşümleri
#1
def swirled(image, map_args={}, output_shape=None, order=1, mode='constant', cval=0.0, clip=True, preserve_range=False):
    #image = imageGlobal
    swirled = swirl(image, rotation=0, strength=10, radius=120)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                   sharex=True, sharey=True)

    ax0.imshow(image, cmap=plt.cm.gray)
    ax0.axis('off')
    ax1.imshow(swirled, cmap=plt.cm.gray)
    ax1.axis('off')

    plt.show()


def inverse_map(args):
    pass


#2
def warp_censure(image, map_args={}, output_shape=None, order=1, mode='constant', cval=0.0, clip=True,
                 preserve_range=False):

    #img_orig = imageGlobal
    tform = tf.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
                               translation=(150, -200))
    img_warp = tf.warp(image, tform)

    detector = CENSURE()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    detector.detect(image)

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[0].set_title("Original Image")

    detector.detect(img_warp)

    ax[1].imshow(img_warp, cmap=plt.cm.gray)
    ax[1].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
                  2 ** detector.scales, facecolors='none', edgecolors='r')
    ax[1].set_title('Transformed Image')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def inverse_map(args):
    pass


#3
def warp_basics(image, map_args={}, output_shape=None, order=1, mode='constant', cval=0.0, clip=True,
                 preserve_range=False):
    #text = imageGlobal

    tform = tf.SimilarityTransform(scale=1, rotation=math.pi / 4,
                                   translation=(image.shape[0] / 2, -100))

    rotated = tf.warp(image, tform)
    back_rotated = tf.warp(rotated, tform.inverse)

    fig, ax = plt.subplots(nrows=3)

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[1].imshow(rotated, cmap=plt.cm.gray)
    ax[2].imshow(back_rotated, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

def inverse_map(args):
    pass


#4
def forward_iradon(radon_image, theta=None, output_size=None, filter='ramp', interpolation='linear', circle=True):
    #image = imageGlobal
    image = shepp_logan_phantom()
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    fig.tight_layout()
    plt.show()
    
#5
def warp_piecewise_affine(image, map_args={}, output_shape=None, order=1, mode='constant', cval=0.0, clip=True, preserve_range=False):
    #image = imageGlobal
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] - 1.5 * 50
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))

    fig, ax = plt.subplots()
    ax.imshow(out)
    ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    ax.axis((0, out_cols, out_rows, 0))
    plt.show()


def inverse_map(args):
    pass

#Morfolojik İşlemler
#erosion (1)
def plot_comparison(orig_phantom, eroded, param):
    pass


def white_tophat_erosion(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    selem = disk(6)
    eroded = erosion(orig_phantom, selem)
    plot_comparison(orig_phantom, eroded, 'erosion')
    plt.show()

#DILATION(2)
def white_tophat_dilation(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    dilated = dilation(orig_phantom, selem)
    plot_comparison(orig_phantom, dilated, 'dilation')
    plt.show()


#white_tophat(3)
def white_tophat_(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    phantom = orig_phantom.copy()
    phantom[340:350, 200:210] = 255
    phantom[100:110, 200:210] = 0

    w_tophat = white_tophat(phantom, selem)
    plot_comparison(phantom, w_tophat, 'white tophat')
    plt.show()

#black_tophat(4)
def black_tophat_(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    phantom = orig_phantom.copy()
    phantom[340:350, 200:210] = 255
    phantom[100:110, 200:210] = 0

    b_tophat = black_tophat(phantom, selem)
    plot_comparison(phantom, b_tophat, 'black tophat')
    plt.show()



#closing5
def closing(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    phantom = orig_phantom.copy()
    phantom[340:350, 200:210] = 255
    phantom[100:110, 200:210] = 0

    phantom = orig_phantom.copy()
    phantom[10:30, 200:210] = 0

    closed = closing(phantom, selem)
    plot_comparison(phantom, closed, 'closing')

    plt.show()


#skeletonize6
def skeletonize_(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    phantom = orig_phantom.copy()
    phantom[340:350, 200:210] = 255
    phantom[100:110, 200:210] = 0

    phantom = orig_phantom.copy()
    phantom[10:30, 200:210] = 0

    sk = skeletonize(image == 0)
    plot_comparison(image, sk, 'skeletonize')
    plt.show()

#convex hull7
def convex_hull(image, selem=None, out=None):
    orig_phantom = img_as_ubyte(image)
    fig, ax = plt.subplots()
    ax.imshow(orig_phantom, cmap=plt.cm.gray)

    def plot_comparison(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')

    phantom = orig_phantom.copy()
    phantom[340:350, 200:210] = 255
    phantom[100:110, 200:210] = 0

    phantom = orig_phantom.copy()
    phantom[10:30, 200:210] = 0

    #img = imageGlobal
    hull1 = convex_hull_image(image == 0)
    plot_comparison(image, hull1, 'convex hull')
    plt.show()

#8
def local_gradient(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False):
    image = img_as_ubyte(image)

    denoised = rank.median(image, disk(2))

    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = rank.gradient(denoised, disk(2))

    labels = watershed(gradient, markers)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original")

    ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Local Gradient")

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()
#9
def markers(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False):
    image = img_as_ubyte(image)

    denoised = rank.median(image, disk(2))

    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = rank.gradient(denoised, disk(2))

    labels = watershed(gradient, markers)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original")

    ax[1].imshow(markers, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Markers")

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()

#10
def segmented(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False):
    image = img_as_ubyte(image)

    denoised = rank.median(image, disk(2))

    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = rank.gradient(denoised, disk(2))

    labels = watershed(gradient, markers)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original")

    ax[1].imshow(image, cmap=plt.cm.gray)
    ax[1].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.7)
    ax[1].set_title("Segmented")

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()
#11
def maksimum_tree(image):

    image = color.rgb2gray(image)

    max_tree = max_tree_local_maxima(image)

    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(8, 8))

    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(max_tree, cmap='gray')
    ax[1].set_title("Max tree ")


    plt.tight_layout()
    plt.show()

#Video İşleme
def video():
    cap = cv2.VideoCapture(0)

    while (1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

#Active Contour Örneği
def active_contour(image):
    def store_evolution_in(lst):
        
        def _store(x):
            lst.append(np.copy(x))
    
        return _store
    
    image = img_as_float(data.camera())
    
    init_ls = checkerboard_level_set(image.shape, 6)
    
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()
    
    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)
    
    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 35")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)
    
    
    
    image = img_as_float(data.coins())
    gimage = inverse_gaussian_gradient(image)
    
    
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                               smoothing=1, balloon=-1,
                                               threshold=0.69,
                                               iter_callback=callback)
    
    ax[2].imshow(image, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("Morphological GAC segmentation", fontsize=12)
    
    ax[3].imshow(ls, cmap="gray")
    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 230")
    ax[3].legend(loc="upper right")
    title = "Morphological GAC evolution"
    ax[3].set_title(title, fontsize=12)
    
    fig.tight_layout()
    plt.show()


root = Root()
root.mainloop()