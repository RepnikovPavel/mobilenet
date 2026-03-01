import cv2
from ImageNetDataset import ImageNetCls

def show_dataset(dataset: ImageNetCls):
    # dataset.eval()
    n = len(dataset)

    idx = 0
    win_name = "ImageNet viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        img, classidx = dataset[idx]
        class_name = dataset.class_index_to_class_name(classidx)

        # копия, чтобы не портить исходное изображение
        vis = img.copy()

        # текст внизу слева
        text = f"{idx}/{n-1}  class: {class_name} ({classidx})"
        cv2.putText(
            vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win_name, vis)

        # ждём нажатие клавиши (0 = бесконечно, можно поставить 50–100 мс)
        key = cv2.waitKey(0) & 0xFF

        # ESC или q — выход
        if key in (27, ord('q')):
            break
        # стрелка вправо или 'd' — следующий
        elif key in (81, 2424832, ord('a')):  # Left, иногда коды разные, см. ниже
            # влево
            idx = (idx - 1) % n
        # стрелка влево или 'a' — предыдущий
        elif key in (83, 2555904, ord('d')):  # Right
            # вправо
            idx = (idx + 1) % n

    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset = ImageNetCls()
    dataset.train()
    print(f"dataset train len {len(dataset)}")
    # dataset.eval()
    # print(f"dataset eval len {len(dataset)}")

    show_dataset(dataset)
