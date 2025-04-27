def extract_license_info(detections):
    """
    Extract emirate, category, and license number from license plate detections
    based on spatial relationships and positioning.

    Args:
        detections: List of detection dictionaries with 'class', 'confidence', and 'bbox' keys

    Returns:
        Dictionary containing emirate, category, and license number
    """
    if not detections:
        return {'emirate': 'Unknown', 'category': 'Unknown', 'plate_number': 'Unknown'}

    # Convert detections to a more convenient format
    elements = []
    for detection in detections:
        if isinstance(detection, dict) and 'bbox' in detection:
            # If detection is already in dictionary format
            x1, y1, x2, y2 = detection['bbox']
            cls = detection['class']
            conf = detection['confidence']
        else:
            # If detection is in the format from your output
            cls = detection[0]
            conf = detection[1]
            x1, y1, x2, y2 = map(int, detection[2].strip('()').split(','))

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        area = width * height

        elements.append({
            'class': cls,
            'confidence': conf,
            'bbox': (x1, y1, x2, y2),
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'area': area
        })

    # First identify the plate itself if it's in the detections
    plate_elements = [
        e for e in elements if 'plate' in str(e['class']).lower()]
    plate_bbox = None
    if plate_elements:
        # Use the largest plate detection
        plate_elements.sort(key=lambda x: x['area'], reverse=True)
        plate_bbox = plate_elements[0]['bbox']
        # Remove plate from further processing
        elements = [e for e in elements if e not in plate_elements]

    # Identify emirate - typically contains "DUBAI" in the class name
    emirate_elements = [e for e in elements if any(em in str(e['class']).upper(
    ) for em in ['DUBAI', 'AJMAN', 'SHARKA', 'ABUDABI', 'FUJIRA', 'RAK', 'AM'])]
    emirate = emirate_elements[0]['class'] if emirate_elements else 'Unknown'

    # Remove emirate from further processing
    if emirate_elements:
        for e in emirate_elements:
            if e in elements:  # Check if element is still in the list
                elements.remove(e)

    # If no elements left after removing emirate, return early
    if not elements:
        return {
            'emirate': emirate,
            'category': 'Unknown',
            'plate_number': 'Unknown'
        }

    # Group elements by their vertical position (y-coordinate)
    # Use a clustering approach to handle slight vertical alignment differences
    y_values = [e['center'][1] for e in elements]
    min_y, max_y = min(y_values), max(y_values)
    vertical_range = max_y - min_y

    # FIX 1: Check for known category letters that are common in UAE plates
    # UAE plates typically use letters like A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, V, W, X, Z
    known_categories = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z'])
    
    # Look for known category letters first
    category_candidates = [e for e in elements if str(e['class']).upper() in known_categories]
    
    if category_candidates:
        # Found explicit category
        category_elements = [category_candidates[0]]  # Use the first detected category
        number_elements = [e for e in elements if e not in category_elements]
        # Sort number elements by x-coordinate
        number_elements.sort(key=lambda e: e['center'][0])
    else:
        # No explicit category detected - handle rows and positioning
        if vertical_range > 20:  # Threshold for multiple rows
            # Group elements by vertical position
            threshold = vertical_range / 2  # Simple threshold for two rows
            median_y = sum(y_values) / len(y_values)

            upper_row = [e for e in elements if e['center'][1] < median_y]
            lower_row = [e for e in elements if e['center'][1] >= median_y]

            # Sort each row by x-coordinate
            upper_row.sort(key=lambda e: e['center'][0])
            lower_row.sort(key=lambda e: e['center'][0])

            # For Dubai plates, typically upper row is category and lower row is number
            if len(upper_row) < len(lower_row) and len(upper_row) <= 2:
                # FIX 2: Check if upper row looks like a valid category
                if any(str(e['class']).isalpha() for e in upper_row):
                    category_elements = upper_row
                    number_elements = lower_row
                else:
                    # Upper row doesn't look like a category - mark as unknown
                    category_elements = []
                    number_elements = elements
                    number_elements.sort(key=lambda e: e['center'][0])
            else:
                # If rows are similar in length or upper is longer, use character type heuristics
                # Look for alphabetic characters that could indicate category
                alpha_elements = [e for e in elements if isinstance(
                    e['class'], str) and e['class'].isalpha()]
                if alpha_elements:
                    # Find the leftmost alphabetic element
                    alpha_elements.sort(key=lambda e: e['center'][0])
                    category_elements = [alpha_elements[0]]
                    number_elements = [
                        e for e in elements if e not in category_elements]
                    number_elements.sort(key=lambda e: e['center'][0])
                else:
                    # FIX 3: Don't default to first element as category if it's not alphabetic
                    category_elements = []
                    number_elements = elements
                    number_elements.sort(key=lambda e: e['center'][0])
        else:
            # Single row plate - sort horizontally
            elements.sort(key=lambda e: e['center'][0])

            # FIX 4: Check if first element is a valid alphabetic category
            if elements and isinstance(elements[0]['class'], str) and elements[0]['class'].upper() in known_categories:
                category_elements = [elements[0]]
                number_elements = elements[1:]
            else:
                # Calculate horizontal gaps between adjacent elements
                gaps = []
                gap_indices = []
                for i in range(1, len(elements)):
                    gap = elements[i]['center'][0] - elements[i-1]['center'][0]
                    gaps.append(gap)
                    gap_indices.append(i-1)

                # Find significant gaps (much larger than the average)
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    significant_gap_indices = [i for i, g in zip(
                        gap_indices, gaps) if g > 1.5 * avg_gap]

                    if significant_gap_indices:
                        # Use the most significant gap to split category and number
                        max_gap_idx = max(significant_gap_indices,
                                        key=lambda i: gaps[i])
                        left_elements = elements[:max_gap_idx+1]
                        right_elements = elements[max_gap_idx+1:]
                        
                        # FIX 5: Only use left elements as category if they're alphabetic
                        if all(isinstance(e['class'], str) and e['class'].isalpha() for e in left_elements):
                            category_elements = left_elements
                            number_elements = right_elements
                        else:
                            # Left elements don't look like a category - mark as unknown
                            category_elements = []
                            number_elements = elements
                    else:
                        # Check for leftmost element being alphabetic (common for category)
                        if isinstance(elements[0]['class'], str) and elements[0]['class'].isalpha():
                            category_elements = [elements[0]]
                            number_elements = elements[1:]
                        else:
                            # FIX 6: Don't default to first element as category if it's not alphabetic
                            category_elements = []
                            number_elements = elements
                else:
                    # FIX 7: Only use first element as category if it's alphabetic
                    if elements and isinstance(elements[0]['class'], str) and elements[0]['class'].isalpha():
                        category_elements = [elements[0]]
                        number_elements = elements[1:] if len(elements) > 1 else []
                    else:
                        category_elements = []
                        number_elements = elements

    # Extract category and plate number as strings
    if category_elements:
        category = ''.join([str(e['class']) for e in category_elements])
    else:
        # FIX 8: Set category to Unknown if no valid category elements were found
        category = 'Unknown'
    
    plate_number = ''.join([str(e['class']) for e in number_elements])

    # FIX 9: Modified post-processing to preserve fully numeric license numbers
    # Only split if we have alphabetic characters that are valid categories
    if category == 'Unknown' and len(plate_number) > 3:
        # Try to find category letters at the beginning
        for i, char in enumerate(plate_number):
            if char.upper() in known_categories:
                # This might be a category letter
                if i == 0:  # If it's the first character
                    category = plate_number[0]
                    plate_number = plate_number[1:]
                    break

    # Debug info that can help analyze the results
    print("Debug Information:")
    print(f"All elements: {[(e['class'], e['center']) for e in elements]}")
    print(f"Emirate elements: {emirate_elements}")
    print(
        f"Category elements: {[(e['class'], e['center']) for e in category_elements]}")
    print(
        f"Number elements: {[(e['class'], e['center']) for e in number_elements]}")

    return {
        'emirate': emirate,
        'category': category,
        'plate_number': plate_number
    }