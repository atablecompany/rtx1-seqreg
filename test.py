# This script is used to test the fundus.py functions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fundus
import time
import skimage

#%% Import video file
# video_path = "G:\PapyrusSorted\AVINA ZAVALA_Marta Ester_19860214_FEMALE\OD_20240405143339\OD_20240405143339_X2.0N_Y0.0_Z0.0_AVINA ZAVALA_Marta Ester_451.mpg"
# video_path = "G:\PapyrusSorted\ABADZHIEVA_Polina_19940415_FEMALE\OD_20240506114845\OD_20240506114845_X2.0N_Y0.0_Z0.0_ABADZHIEVA_Polina_496.mpg"
# video_path = "G:\PapyrusSorted\FRICKER_Isabelle_19880404_FEMALE\OD_20231123140522\OD_20231123140522_X0.0T_Y-2.0_Z0.0_FRICKER_Isabelle_007.mpg"
# video_path = "G:\PapyrusSorted\BECK_Hanna_19870504_FEMALE\OD_20240327091635\OD_20240327091635_X0.0T_Y2.0_Z30.0_BECK_Hanna_192.mpg"

# Nektere pripady
# video_path = "G:\PapyrusSorted\BUECH_Eva_19811017_FEMALE\OD_20231215124843\OD_20231215124843_X2.0N_Y0.0_Z0.0_BUECH_Eva_237.mpg"  # Koregistrovalo to různý oblasti
# video_path = "G:\PapyrusSorted\FRANZ_Juliane_19910412_FEMALE\OD_20231220104415\OD_20231220104415_X13.7N_Y4.9_Z170.0_FRANZ_Juliane_271.mpg"  # Koregistrovalo to různý oblasti
# video_path = "G:\PapyrusSorted\BOERNER_Carolin_20050614_FEMALE\OD_20240104122715\OD_20240104122715_X10.5N_Y8.5_Z120.0_BOERNER_Carolin_311.mpg"  # Koregistrovalo to různý oblasti
# TODO: Co s tím? Když to registruje různý oblasti?
# video_path = "G:\PapyrusSorted\BUEGIEL_Patricia_19860111_FEMALE\OD_20240215124239\OD_20240215124239_X2.0T_Y0.0_Z0.0_BUEGIEL_Patricia_354.mpg"  # Většina outlierů ale vypadá takhle (není tam nic, jenom šum)
# video_path = "G:\PapyrusSorted\FARR_Anja_19850912_FEMALE\OD_20231219102331\OD_20231219102331_X10.8N_Y8.3_Z150.0_FARR_Anja_295.mpg"  # Tady je vysledek straightup horší než reference
# video_path = "G:\PapyrusSorted\BUECH_Eva_19811017_FEMALE\OD_20231215125017\OD_20231215125017_X11.6N_Y-6.8_Z160.0_BUECH_Eva_237.mpg"  # Tady je moc velkej pohyb a tak to špatně koregistrovalo
# video_path = "G:\PapyrusSorted\CIKAJ_Ajne_19870102_FEMALE\OD_20231120112629\OD_20231120112629_X10.4N_Y8.1_Z140.0_CIKAJ_Ajne_235.mpg"  # Špatnej ořez (rozšíření) vkůli černýmu pruhu vpravo
video_path = "G:\PapyrusSorted\FREIER_Sophie_19901012_FEMALE\OD_20240523094753\OD_20240523094753_X0.0T_Y2.0_Z-10.0_FREIER_Sophie_213.mpg"  # Tohle by možná šlo líp kdyby to vzalo víc snímků
# video_path = "G:\PapyrusSorted\AS-II_Aiza_19850619_FEMALE\OD_20231218140208\OD_20231218140208_X0.0T_Y-2.0_Z0.0_AS-II_Aiza_NS014.mpg"  # Tohle by možná šlo líp kdyby to vzalo víc snímků
# video_path = "G:\PapyrusSorted\AL-EESO_Nariman Salman Khalaf_19900324_FEMALE\OD_20230608144231\OD_20230608144231_X0.0T_Y-2.0_Z30.0_AL-EESO_Nariman Salman Khalaf_P017.mpg"  # Tohle by možná šlo líp kdyby to vzalo víc snímků
# video_path = "G:\PapyrusSorted\BUKWITSCHKA_Tina_19910306_FEMALE\OD_20230430132545\OD_20230430132545_X0.2N_Y-0.6_Z0.0_BUKWITSCHKA_Tina.mpg"  # Proč to vzalo jen 2 snímky??
# video_path = "G:\PapyrusSorted\GRASSMANN_Antje_19930614_FEMALE\OD_20230802130310\OD_20230802130310_X0.0T_Y0.0_Z30.0_GRASSMANN_Antje_P031.mpg"  # Vzalo to málo snímků
# video_path = "G:\PapyrusSorted\ALLRICH_Ina_19870402_FEMALE\OD_20240605110908\OD_20240605110908_X0.0T_Y2.0_Z10.0_ALLRICH_Ina_520.mpg"  # Tohle by možná šlo líp kdyby to vzalo víc snímků
# video_path = "G:\PapyrusSorted\CAMPOS ZACARIAS_Osmarze del_19980810_FEMALE\OD_20231109140323\OD_20231109140323_X8.3N_Y-6.0_Z120.0_CAMPOS ZACARIAS_Osmarze del_207.mpg"  # Tohle by možná šlo líp kdyby to vzalo víc snímků
# video_path = "G:\PapyrusSorted\AL-EESO_Nariman Salman Khalaf_19900324_FEMALE\OD_20230531174510\OD_20230531174510_X2.0T_Y0.0_Z30.0_AL-EESO_Nariman Salman Khalaf_P017.mpg"  # Tohle by možná šlo líp kdyby to vzalo víc snímků
# TODO: Co s tím? Když to bere málo snímků?


reference_path = video_path.replace(".mpg", ".png")
report_path = video_path.replace(".mpg", "_report_test.txt")
start_time = time.time()
frames = fundus.load_video(video_path)
reference = fundus.load_reference_image(reference_path)
print(frames.shape)

#%% Determine the sharpness of frames
# metric = 'loc_var_of_gray'
sharpness = fundus.calculate_sharpness(frames)
selected_frames = fundus.select_frames(frames, sharpness)

# sharpness_threshold = 0.8
# selected_frames = fundus.select_frames(frames, sharpness, threshold=sharpness_threshold)

#%% Show individual frames
# for i in range(len(frames)):
#     fundus.show_frame(frames[i], sharpness=sharpness[i], custom_note=i)

#%% Plot sharpness over frames
plt.figure(figsize=(10, 6))
plt.plot(sharpness, marker='o', linestyle='-', color='b')
plt.title(f'Sharpness over frames (avg)')
plt.xlabel('Frame index')
plt.ylabel('Sharpness')
plt.grid(True)
# plt.savefig('sharpness_plot.svg', format='svg')
plt.show()

#%% Perform registration
reg = fundus.register(selected_frames, sharpness, reference='mean', pad='same')
# TODO: Register pomoci elastixu v podstate vubec nefunguje
# for i in range(len(reg)):
#     fundus.save_frame(reg[i], f"C:/Users/tengl/PycharmProjects/dp/reg/frame_{i}.png")

#%% Export registered frames as lossless avi
output_path = "C:/Users/tengl/PycharmProjects/dp/registered_frames.avi"
height, width = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Lossless codec
out = cv2.VideoWriter(output_path, fourcc, 15, (width, height), False)

for frame in reg:
    # Ensure frame is in the correct format (uint8)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    out.write(frame)

out.release()
print(f"Video saved to {output_path}")

#%% Cumulate registered frames
cum = fundus.cumulate(reg, method='mean')
# fundus.save_frame(cum, "C:/Users/tengl/PycharmProjects/dp/pokus/cum_wt3.png")
# fundus.assess_quality(cum, "C:/Users/tengl/PycharmProjects/dp/pokus/cum_wt3.txt")

#%% Show result
fundus.show_frame(cum)
fundus.show_frame(reference, custom_note="Reference image\n")

#%% Apply additional denoising
# weight = 2
denoised = fundus.denoise(cum)
fundus.show_frame(denoised, custom_note=f"Denoised\n")

#%% Assess quality
elapsed_time = time.time() - start_time
sharpness_log, brisque, piqe, niqe = fundus.assess_quality(cum, report_path)
print("Processing took: {:.2f} seconds".format(elapsed_time))
print("Sharpness LoG Image: ", sharpness_log[0])
print("Sharpness LoG Reference: ", sharpness_log[1])
print("BRISQUE Image: ", brisque[0])
print("BRISQUE Reference: ", brisque[1])
print("PIQE Image: ", piqe[0])
print("PIQE Reference: ", piqe[1])
print("NIQE Image: ", niqe[0])
print("NIQE Reference: ", niqe[1])

