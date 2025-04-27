def filter_overlapping_detections(elements, iou_threshold=0.5, center_distance_threshold=10):
    """
    Filter out overlapping detections, keeping only the highest confidence detection for each overlapping group.

    Args:
        elements: List of detection dictionaries
        iou_threshold: Threshold for IoU (Intersection over Union) to determine overlap
        center_distance_threshold: Threshold for center distance (in pixels) to determine overlap

    Returns:
        Filtered list of detections with overlaps removed
    """
    if not elements:
        return []

    # Sort elements by confidence in descending order
    sorted_elements = sorted(
        elements, key=lambda x: x['confidence'], reverse=True)

    # Initialize list of filtered elements with the highest confidence element
    filtered_elements = [sorted_elements[0]]

    # Check each element against the filtered list
    for element in sorted_elements[1:]:
        is_overlapping = False

        for filtered_element in filtered_elements:
            # Don't consider elements overlapping if they are both digits and horizontally adjacent
            if (str(element['class']).isdigit() and str(filtered_element['class']).isdigit()):
                # Calculate horizontal distance (more lenient for digits)
                horizontal_distance = abs(
                    element['center'][0] - filtered_element['center'][0])

                # If they're far enough apart horizontally, don't consider them overlapping
                if horizontal_distance >= element['width'] * 0.5:
                    continue

            # Check if the centers are close
            center_distance_squared = (element['center'][0] - filtered_element['center'][0]) ** 2 + \
                                      (element['center'][1] -
                                       filtered_element['center'][1]) ** 2

            if center_distance_squared <= center_distance_threshold ** 2:
                is_overlapping = True
                break

            # If centers are not close, check IOU
            if not is_overlapping:
                # Calculate IOU
                # Get coordinates
                x1_a, y1_a, x2_a, y2_a = element['bbox']
                x1_b, y1_b, x2_b, y2_b = filtered_element['bbox']

                # Calculate intersection area
                x_left = max(x1_a, x1_b)
                y_top = max(y1_a, y1_b)
                x_right = min(x2_a, x2_b)
                y_bottom = min(y2_a, y2_b)

                if x_right < x_left or y_bottom < y_top:
                    intersection_area = 0
                else:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)

                # Calculate union area
                area_a = element['area']
                area_b = filtered_element['area']
                union_area = area_a + area_b - intersection_area

                # Calculate IOU
                iou = intersection_area / union_area if union_area > 0 else 0

                if iou > iou_threshold:
                    is_overlapping = True
                    break

        # If this element doesn't overlap with any element in the filtered list, add it
        if not is_overlapping:
            filtered_elements.append(element)

    # Sort filtered elements back by their center x-coordinate
    filtered_elements.sort(key=lambda e: e['center'][0])

    print(
        f"Filtered out {len(elements) - len(filtered_elements)} overlapping detections")

    return filtered_elements
