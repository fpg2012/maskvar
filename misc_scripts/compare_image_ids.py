from itertools import islice

from maskvar.maskseg_build_everything import (
    build_cocolvis_dataset,
    build_coconut_hf_dataset,
)

if __name__ == '__main__':
    cocolvis_train, cocolvis_val = build_cocolvis_dataset()
    coconut_train, coconut_val = build_coconut_hf_dataset()

    image_id_lvis = []
    iamge_id_coconut = []

    for item in islice(cocolvis_train, 200):
        img, mask, instinfo, image_id = item
        image_id_lvis.append(image_id)
    
    for item in islice(coconut_val, 200):
        img, mask, instinfo, image_id = item
        iamge_id_coconut.append(image_id)

    print("COCO-LVIS image_ids:", image_id_lvis)
    print("COCONut image_ids:", iamge_id_coconut)

    print("dataset_size:")
    print("COCO-LVIS:", len(cocolvis_train))
    print("COCONut:", len(coconut_train))