import os
import csv
import xml.etree.ElementTree as ET

folder_path = "D:/MVS3DM/xml/"
output_csv = "D:/MVS3DM_data.csv"
data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".XML"):
        file_path = os.path.join(folder_path, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        filename_tag = root.find('.//FILENAME').text if root.find('.//FILENAME') is not None else 'N/A'
        scan_direction = root.find('.//SCANDIRECTION').text if root.find('.//SCANDIRECTION') is not None else 'N/A'
        tlc_time = root.find('.//TLCTIME').text if root.find('.//TLCTIME') is not None else 'N/A'
        mean_sun_az = root.find('.//MEANSUNAZ').text if root.find('.//MEANSUNAZ') is not None else 'N/A'
        mean_sun_el = root.find('.//MEANSUNEL').text if root.find('.//MEANSUNEL') is not None else 'N/A'
        mean_sat_az = root.find('.//MEANSATAZ').text if root.find('.//MEANSATAZ') is not None else 'N/A'
        mean_sat_el = root.find('.//MEANSATEL').text if root.find('.//MEANSATEL') is not None else 'N/A'
        mean_intrack_view_angle = root.find('.//MEANINTRACKVIEWANGLE').text if root.find('.//MEANINTRACKVIEWANGLE') is not None else 'N/A'
        mean_crosstrack_view_angle = root.find('.//MEANCROSSTRACKVIEWANGLE').text if root.find('.//MEANCROSSTRACKVIEWANGLE') is not None else 'N/A'
        mean_off_nadir_view_angle = root.find('.//MEANOFFNADIRVIEWANGLE').text if root.find('.//MEANOFFNADIRVIEWANGLE') is not None else 'N/A'
        data.append([filename_tag, scan_direction, tlc_time, mean_sun_az, mean_sun_el, mean_sat_az, mean_sat_el, mean_intrack_view_angle, mean_crosstrack_view_angle, mean_off_nadir_view_angle])

with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['FILENAME', 'SCANDIRECTION', 'TLCTIME', 'MEANSUNAZ', 'MEANSUNEL','MEANSATAZ','MEANSATEL', 'MEANINTRACKVIEWANGLE', 'MEANCROSSTRACKVIEWANGLE', 'MEANOFFNADIRVIEWANGLE'])
    writer.writerows(data)

print(f"Data has been successfully written to {output_csv}")




