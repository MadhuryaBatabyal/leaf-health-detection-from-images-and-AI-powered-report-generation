import cv2


def draw_leaf_box(image, leaf_bbox):
    output = image.copy()

    if isinstance(leaf_bbox, tuple):
        x1, y1, x2, y2 = leaf_bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            "Leaf",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


def draw_pests(image, pest_result):
    output = image.copy()

    if isinstance(pest_result, dict) and "pests" in pest_result:
        for pest in pest_result["pests"]:
            x1, y1, x2, y2 = pest["bbox"]
            label = pest["label"]
            conf = pest["confidence"]

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                output,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return output


def draw_disease_label(image, disease_result):
    output = image.copy()

    if isinstance(disease_result, dict) and "label" in disease_result:
        label = disease_result["label"]
        conf = disease_result["confidence"]

        text = f"Disease: {label} ({conf:.2f})"

        cv2.putText(
            output,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    return output
