import numpy as np
import matplotlib.pyplot as plt

def save_image(noise_intensities, success_rates,fiilepath):
    plt.plot(noise_intensities, success_rates, marker='o')
    plt.xlabel('Noise Intensity(Std)')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs. Noise Intensity')
    plt.savefig(fiilepath)
    #plt.show()

noise_intensities=[0,0.01,0.05,0.1,0.15,0.2]
success_rates=[1,1,0.2,0.25,0.05,0.05]
fiilepath='/research/d1/rshr/arlin/data/debug/depth_noise_debug/noise-sr-seg-d.png'
save_image(noise_intensities, success_rates,fiilepath)