from vision_agent.agent.vision_agent_coder import strip_function_calls


def test_strip_non_function_real_case():
    code = """import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool


from vision_agent.tools import load_image, owl_v2_image, overlay_bounding_boxes, save_image, save_json

def check_helmets(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Detect people and helmets
    detections = owl_v2_image("person, helmet", image, box_threshold=0.2)
    
    # Separate people and helmets
    people = [d for d in detections if d['label'] == 'person']
    helmets = [d for d in detections if d['label'] == 'helmet']
    
    people_with_helmets = 0
    people_without_helmets = 0
    
    height, width = image.shape[:2]
    
    for person in people:
        person_x = (person['bbox'][0] + person['bbox'][2]) / 2
        person_y = person['bbox'][1]  # Top of the bounding box
        
        helmet_found = False
        for helmet in helmets:
            helmet_x = (helmet['bbox'][0] + helmet['bbox'][2]) / 2
            helmet_y = (helmet['bbox'][1] + helmet['bbox'][3]) / 2
            
            # Check if the helmet is within 20 pixels of the person's head
            if (abs((helmet_x - person_x) * width) < 20 and
                -5 < ((helmet_y - person_y) * height) < 20):
                helmet_found = True
                break
        
        if helmet_found:
            people_with_helmets += 1
            person['label'] = 'person with helmet'
        else:
            people_without_helmets += 1
            person['label'] = 'person without helmet'
    
    # Create the count dictionary
    count_dict = {
        "people_with_helmets": people_with_helmets,
        "people_without_helmets": people_without_helmets
    }
    
    # Visualize the results
    visualized_image = overlay_bounding_boxes(image, detections)
    
    # Save the visualized image
    save_image(visualized_image, "/home/user/visualized_result.png")
    
    # Save the count dictionary as JSON
    save_json(count_dict, "/home/user/helmet_counts.json")
    
    return count_dict

# The function can be called with the image path
result = check_helmets("/home/user/edQPXGK_workers.png")"""
    expected_code = """import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool


from vision_agent.tools import load_image, owl_v2_image, overlay_bounding_boxes, save_image, save_json

def check_helmets(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Detect people and helmets
    detections = owl_v2_image("person, helmet", image, box_threshold=0.2)
    
    # Separate people and helmets
    people = [d for d in detections if d['label'] == 'person']
    helmets = [d for d in detections if d['label'] == 'helmet']
    
    people_with_helmets = 0
    people_without_helmets = 0
    
    height, width = image.shape[:2]
    
    for person in people:
        person_x = (person['bbox'][0] + person['bbox'][2]) / 2
        person_y = person['bbox'][1]  # Top of the bounding box
        
        helmet_found = False
        for helmet in helmets:
            helmet_x = (helmet['bbox'][0] + helmet['bbox'][2]) / 2
            helmet_y = (helmet['bbox'][1] + helmet['bbox'][3]) / 2
            
            # Check if the helmet is within 20 pixels of the person's head
            if (abs((helmet_x - person_x) * width) < 20 and
                -5 < ((helmet_y - person_y) * height) < 20):
                helmet_found = True
                break
        
        if helmet_found:
            people_with_helmets += 1
            person['label'] = 'person with helmet'
        else:
            people_without_helmets += 1
            person['label'] = 'person without helmet'
    
    # Create the count dictionary
    count_dict = {
        "people_with_helmets": people_with_helmets,
        "people_without_helmets": people_without_helmets
    }
    
    # Visualize the results
    visualized_image = overlay_bounding_boxes(image, detections)
    
    # Save the visualized image
    save_image(visualized_image, "/home/user/visualized_result.png")
    
    # Save the count dictionary as JSON
    save_json(count_dict, "/home/user/helmet_counts.json")
    
    return count_dict

# The function can be called with the image path"""
    code_out = strip_function_calls(code, exclusions=["register_heif_opener"])
    assert code_out == expected_code
