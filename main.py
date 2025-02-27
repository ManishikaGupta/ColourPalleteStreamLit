import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# Function to classify the undertone of the dominant color
def classify_undertone(rgb_color):
    r, g, b = rgb_color
    if r > g and g > b:
        return "Warm"
    elif b > r and b > g:
        return "Cool"
    else:
        return "Neutral"

# Function to find the dominant skin tone from the image
def find_dominant_skin_tone(image, n_clusters=3):
    # Convert image from PIL format to a numpy array in RGB
    image_array = np.array(image)

    # Reshape the image to a 2D array of pixels
    pixel_data = image_array.reshape((-1, 3))

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_data)

    # Get the cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Identify the dominant skin tone cluster
    labels = kmeans.labels_
    skin_cluster = np.argmax(np.bincount(labels))

    # Get the dominant skin tone color
    skin_tone_color = dominant_colors[skin_cluster]

    # Classify the skin undertone
    undertone = classify_undertone(skin_tone_color)

    return undertone

# DATASET
dataset = [
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Black",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Olive Green", "RGB": (107, 142, 35)},
            {"Color Name": "Mustard", "RGB": (255, 219, 88)},
            {"Color Name": "Terracotta", "RGB": (226, 114, 91)},
            {"Color Name": "Deep Red", "RGB": (139, 0, 0)},
            {"Color Name": "Burnt Orange", "RGB": (204, 85, 0)},
            {"Color Name": "Warm Taupe", "RGB": (150, 111, 51)},
            {"Color Name": "Coral", "RGB": (255, 127, 80)},
            {"Color Name": "Copper", "RGB": (184, 115, 51)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Black",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Coral", "RGB": (255, 127, 80)},
            {"Color Name": "Peach", "RGB": (255, 218, 185)},
            {"Color Name": "Warm Green", "RGB": (85, 107, 47)},
            {"Color Name": "Copper", "RGB": (184, 115, 51)},
            {"Color Name": "Rust", "RGB": (183, 65, 14)},
            {"Color Name": "Amber", "RGB": (255, 191, 0)},
            {"Color Name": "Honey", "RGB": (255, 186, 120)},
            {"Color Name": "Warm Grey", "RGB": (181, 166, 134)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Black",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Warm Brown", "RGB": (139, 69, 19)},
            {"Color Name": "Soft Pink", "RGB": (255, 182, 193)},
            {"Color Name": "Mint Green", "RGB": (152, 255, 152)},
            {"Color Name": "Gold", "RGB": (255, 215, 0)},
            {"Color Name": "Warm Mauve", "RGB": (153, 102, 102)},
            {"Color Name": "Champagne", "RGB": (250, 235, 215)},
            {"Color Name": "Rose", "RGB": (255, 0, 127)},
            {"Color Name": "Caramel", "RGB": (210, 105, 30)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Black",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Warm Orange", "RGB": (255, 140, 0)},
            {"Color Name": "Golden Yellow", "RGB": (255, 223, 0)},
            {"Color Name": "Peach", "RGB": (255, 218, 185)},
            {"Color Name": "Earthy Brown", "RGB": (150, 75, 0)},
            {"Color Name": "Light Teal", "RGB": (144, 200, 180)},
            {"Color Name": "Apricot", "RGB": (251, 206, 177)},
            {"Color Name": "Coral Pink", "RGB": (248, 131, 121)},
            {"Color Name": "Warm Yellow", "RGB": (252, 209, 22)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Brown",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Deep Orange", "RGB": (255, 140, 0)},
            {"Color Name": "Olive Green", "RGB": (107, 142, 35)},
            {"Color Name": "Warm Red", "RGB": (165, 42, 42)},
            {"Color Name": "Caramel", "RGB": (210, 105, 30)},
            {"Color Name": "Warm Beige", "RGB": (245, 245, 220)},
             {"Color Name": "Bronze", "RGB": (205, 127, 50)},
            {"Color Name": "Ochre", "RGB": (204, 119, 34)},
            {"Color Name": "Saffron", "RGB": (244, 196, 48)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Brown",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Coral", "RGB": (255, 127, 80)},
            {"Color Name": "Light Pink", "RGB": (255, 182, 193)},
            {"Color Name": "Terracotta", "RGB": (226, 114, 91)},
            {"Color Name": "Warm Green", "RGB": (85, 107, 47)},
            {"Color Name": "Copper", "RGB": (184, 115, 51)},
            {"Color Name": "Peach", "RGB": (255, 203, 164)},
            {"Color Name": "Burnt Sienna", "RGB": (233, 116, 81)},
            {"Color Name": "Mocha", "RGB": (192, 115, 56)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Brown",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Soft Coral", "RGB": (240, 128, 128)},
            {"Color Name": "Golden Brown", "RGB": (153, 101, 21)},
            {"Color Name": "Pastel Yellow", "RGB": (253, 253, 150)},
            {"Color Name": "Rust", "RGB": (183, 65, 14)},
            {"Color Name": "Amber", "RGB": (255, 191, 0)},
            {"Color Name": "Sandy Brown", "RGB": (244, 164, 96)},
            {"Color Name": "Cinnamon", "RGB": (123, 63, 0)},
            {"Color Name": "Antique White", "RGB": (250, 235, 215)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Brown",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Salmon", "RGB": (250, 128, 114)},
            {"Color Name": "Goldenrod", "RGB": (218, 165, 32)},
            {"Color Name": "Soft Teal", "RGB": (144, 200, 180)},
            {"Color Name": "Copper", "RGB": (184, 115, 51)},
            {"Color Name": "Light Coral", "RGB": (240, 128, 128)},
            {"Color Name": "Soft Yellow", "RGB": (255, 255, 204)},
            {"Color Name": "Honey Yellow", "RGB": (250, 214, 165)},
            {"Color Name": "Dusty Rose", "RGB": (205, 92, 92)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Blonde",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Burnt Orange", "RGB": (204, 85, 0)},
            {"Color Name": "Khaki", "RGB": (195, 176, 145)},
            {"Color Name": "Coral Pink", "RGB": (248, 131, 121)},
            {"Color Name": "Sage Green", "RGB": (188, 184, 138)},
            {"Color Name": "Light Brown", "RGB": (181, 101, 29)},
            {"Color Name": "Butterscotch", "RGB": (255, 182, 82)},
            {"Color Name": "Tan", "RGB": (210, 180, 140)},
            {"Color Name": "Sunset", "RGB": (255, 97, 56)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Blonde",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Peach", "RGB": (255, 218, 185)},
            {"Color Name": "Golden Yellow", "RGB": (255, 223, 0)},
            {"Color Name": "Warm Tan", "RGB": (210, 180, 140)},
            {"Color Name": "Soft Green", "RGB": (152, 251, 152)},
            {"Color Name": "Amber", "RGB": (255, 191, 0)},
             {"Color Name": "Light Salmon", "RGB": (255, 160, 122)},
            {"Color Name": "Golden Honey", "RGB": (255, 221, 148)},
            {"Color Name": "Coral Red", "RGB": (255, 64, 64)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Blonde",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Coral", "RGB": (255, 127, 80)},
            {"Color Name": "Warm Beige", "RGB": (245, 245, 220)},
            {"Color Name": "Pastel Green", "RGB": (119, 221, 119)},
            {"Color Name": "Rust", "RGB": (183, 65, 14)},
            {"Color Name": "Gold", "RGB": (255, 215, 0)},
            {"Color Name": "Light Olive", "RGB": (192, 192, 168)},
            {"Color Name": "Apricot Orange", "RGB": (251, 206, 177)},
            {"Color Name": "Cantaloupe", "RGB": (255, 204, 153)}
        ],
    },
    {
        "Skin Undertone": "Warm",
        "Hair Color": "Blonde",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Light Coral", "RGB": (240, 128, 128)},
            {"Color Name": "Peach", "RGB": (255, 218, 185)},
            {"Color Name": "Soft Yellow", "RGB": (255, 239, 213)},
            {"Color Name": "Mint Green", "RGB": (152, 255, 152)},
            {"Color Name": "Copper", "RGB": (184, 115, 51)},
            {"Color Name": "Warm Blush", "RGB": (255, 192, 203)},
            {"Color Name": "Soft Apricot", "RGB": (255, 189, 164)},
            {"Color Name": "Golden Peach", "RGB": (255, 204, 153)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Black",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Royal Blue", "RGB": (65, 105, 225)},
            {"Color Name": "Silver", "RGB": (192, 192, 192)},
            {"Color Name": "Fuchsia", "RGB": (255, 0, 255)},
            {"Color Name": "Charcoal Grey", "RGB": (54, 69, 79)},
            {"Color Name": "Ice Blue", "RGB": (173, 216, 230)},
            {"Color Name": "Cool Pink", "RGB": (255, 192, 203)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Cobalt", "RGB": (0, 71, 171)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Black",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Burgundy", "RGB": (128, 0, 32)},
            {"Color Name": "Navy Blue", "RGB": (0, 0, 128)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Charcoal Grey", "RGB": (54, 69, 79)},
            {"Color Name": "Plum", "RGB": (142, 69, 133)},
            {"Color Name": "Slate", "RGB": (112, 128, 144)},
            {"Color Name": "Mulberry", "RGB": (197, 75, 140)},
            {"Color Name": "Cool Violet", "RGB": (129, 102, 186)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Black",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Sea Green", "RGB": (46, 139, 87)},
            {"Color Name": "Lilac", "RGB": (200, 162, 200)},
            {"Color Name": "Teal", "RGB": (0, 128, 128)},
            {"Color Name": "Steel Blue", "RGB": (70, 130, 180)},
            {"Color Name": "Dusty Pink", "RGB": (205, 92, 92)},
            {"Color Name": "Winter White", "RGB": (248, 248, 255)},
            {"Color Name": "Frost", "RGB": (219, 225, 228)},
            {"Color Name": "Slate Blue", "RGB": (106, 90, 205)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Black",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Magenta", "RGB": (255, 0, 255)},
            {"Color Name": "Cobalt Blue", "RGB": (0, 71, 171)},
            {"Color Name": "Cool Grey", "RGB": (140, 146, 172)},
            {"Color Name": "Mint", "RGB": (189, 252, 201)},
            {"Color Name": "Rose Pink", "RGB": (255, 102, 204)},
            {"Color Name": "Powder Blue", "RGB": (176, 224, 230)},
            {"Color Name": "Cerulean", "RGB": (0, 123, 167)},
            {"Color Name": "Sky Blue", "RGB": (135, 206, 235)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Brown",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Violet", "RGB": (143, 0, 255)},
            {"Color Name": "Sky Blue", "RGB": (135, 206, 235)},
            {"Color Name": "Turquoise", "RGB": (64, 224, 208)},
            {"Color Name": "Periwinkle", "RGB": (204, 204, 255)},
            {"Color Name": "Mauve", "RGB": (224, 176, 255)},
            {"Color Name": "Lilac Grey", "RGB": (200, 162, 200)},
            {"Color Name": "Slate Grey", "RGB": (112, 128, 144)},
            {"Color Name": "Aubergine", "RGB": (55, 0, 40)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Brown",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Amethyst", "RGB": (153, 102, 204)},
            {"Color Name": "Emerald Green", "RGB": (80, 200, 120)},
            {"Color Name": "Baby Blue", "RGB": (137, 207, 240)},
            {"Color Name": "Rose", "RGB": (255, 0, 127)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Amethyst", "RGB": (153, 102, 204)},
            {"Color Name": "Icy Pink", "RGB": (245, 204, 228)},
            {"Color Name": "Mulberry", "RGB": (197, 75, 140)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Brown",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Blue Grey", "RGB": (102, 153, 204)},
            {"Color Name": "Lavender Blush", "RGB": (255, 240, 245)},
            {"Color Name": "Aqua", "RGB": (127, 255, 212)},
            {"Color Name": "Pale Pink", "RGB": (250, 218, 221)},
            {"Color Name": "Ice Blue", "RGB": (173, 216, 230)},
            {"Color Name": "Dusty Lilac", "RGB": (153, 102, 102)},
            {"Color Name": "Cool Green", "RGB": (144, 238, 144)},
            {"Color Name": "Slate", "RGB": (112, 128, 144)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Brown",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Royal Purple", "RGB": (120, 81, 169)},
            {"Color Name": "Seafoam Green", "RGB": (159, 226, 191)},
            {"Color Name": "Baby Pink", "RGB": (244, 194, 194)},
            {"Color Name": "Sapphire", "RGB": (15, 82, 186)},
            {"Color Name": "Light Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Powder Blue", "RGB": (176, 224, 230)},
            {"Color Name": "Denim", "RGB": (21, 96, 189)},
            {"Color Name": "Electric Blue", "RGB": (125, 249, 255)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Blonde",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Ice Blue", "RGB": (173, 216, 230)},
            {"Color Name": "Cool Pink", "RGB": (255, 192, 203)},
            {"Color Name": "Steel Blue", "RGB": (70, 130, 180)},
            {"Color Name": "Lilac", "RGB": (200, 162, 200)},
            {"Color Name": "Pale Yellow", "RGB": (255, 255, 204)},
            {"Color Name": "Platinum", "RGB": (229, 228, 226)},
            {"Color Name": "Dusty Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Arctic Blue", "RGB": (201, 229, 232)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Blonde",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Mint", "RGB": (189, 252, 201)},
            {"Color Name": "Pale Blue", "RGB": (175, 238, 238)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Plum", "RGB": (142, 69, 133)},
            {"Color Name": "Soft Grey", "RGB": (211, 211, 211)},
            {"Color Name": "Cool Blue", "RGB": (153, 204, 255)},
            {"Color Name": "Amethyst", "RGB": (153, 102, 204)},
            {"Color Name": "Sky Blue", "RGB": (135, 206, 235)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Blonde",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Soft Violet", "RGB": (238, 130, 238)},
            {"Color Name": "Sky Blue", "RGB": (135, 206, 235)},
            {"Color Name": "Pale Mint", "RGB": (152, 255, 152)},
            {"Color Name": "Dusty Rose", "RGB": (205, 92, 92)},
            {"Color Name": "Ice Blue", "RGB": (173, 216, 230)},
            {"Color Name": "Cobalt", "RGB": (0, 71, 171)},
            {"Color Name": "Winterberry", "RGB": (140, 80, 83)},
            {"Color Name": "Pearl", "RGB": (234, 224, 200)}
        ],
    },
    {
        "Skin Undertone": "Cool",
        "Hair Color": "Blonde",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Periwinkle", "RGB": (204, 204, 255)},
            {"Color Name": "Turquoise", "RGB": (64, 224, 208)},
            {"Color Name": "Rose Pink", "RGB": (255, 102, 204)},
            {"Color Name": "Mauve", "RGB": (224, 176, 255)},
            {"Color Name": "Silver", "RGB": (192, 192, 192)},
            {"Color Name": "Silver Mist", "RGB": (207, 210, 218)},
            {"Color Name": "Cool Turquoise", "RGB": (95, 197, 221)},
            {"Color Name": "Icy Peach", "RGB": (255, 203, 203)}
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Black",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Beige", "RGB": (245, 245, 220)},
            {"Color Name": "Soft White", "RGB": (255, 250, 250)},
            {"Color Name": "Taupe", "RGB": (72, 60, 50)},
            {"Color Name": "Olive", "RGB": (128, 128, 0)},
            {"Color Name": "Warm Grey", "RGB": (119, 136, 153)},
            {"Color Name": "Royal Blue", "RGB": (65,105,225)},
            {"Color Name": "Cranberry Red", "RGB": (159,0,15)},
            {"Color Name": "Emerald Green", "RGB": (80,200,120)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Black",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Chocolate Brown", "RGB": (123, 63, 0)},
            {"Color Name": "Ivory", "RGB": (255, 255, 240)},
            {"Color Name": "Camel", "RGB": (193, 154, 107)},
            {"Color Name": "Denim Blue", "RGB": (21, 96, 189)},
            {"Color Name": "Forest Green", "RGB": (34, 139, 34)},
            {"Color Name": "Deep Olive Green", "RGB": (85,107,47)},
            {"Color Name": "Slate Blue", "RGB": (112, 128, 144)},
            {"Color Name": "Burgundy", "RGB": ((128, 0, 32))},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Black",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Soft Olive", "RGB": (192, 194, 105)},
            {"Color Name": "Warm Beige", "RGB": (245, 245, 220)},
            {"Color Name": "Cool Taupe", "RGB": (140, 133, 120)},
            {"Color Name": "Deep Teal", "RGB": (0, 128, 128)},
            {"Color Name": "Pale Yellow", "RGB": (255, 255, 204)},
            {"Color Name": "Slate Gray", "RGB": (112, 128, 144)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Ice Blue", "RGB": (173, 216, 230)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Black",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Warm White", "RGB": (255, 248, 231)},
            {"Color Name": "Navy", "RGB": (0, 0, 128)},
            {"Color Name": "Sage Green", "RGB": (188, 184, 138)},
            {"Color Name": "Light Pink", "RGB": (255, 182, 193)},
            {"Color Name": "Pale Blue", "RGB": (175, 238, 238)},
            {"Color Name": "Navy Blue", "RGB": (0, 0, 128)},
            {"Color Name": "Soft Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Dusty Rose", "RGB": (205, 150, 158)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Brown",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Terracotta", "RGB": (226, 114, 91)},
            {"Color Name": "Moss Green", "RGB": (173, 223, 173)},
            {"Color Name": "Sand", "RGB": (194, 178, 128)},
            {"Color Name": "Slate Blue", "RGB": (106, 90, 205)},
            {"Color Name": "Cream", "RGB": (255, 253, 208)},
            {"Color Name": "Olive Green", "RGB": (107, 142, 35)},
            {"Color Name": "Rust", "RGB": (183, 65, 14)},
            {"Color Name": "Dusty Mauve", "RGB": (145, 95, 109)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Brown",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Mahogany", "RGB": (192, 64, 0)},
            {"Color Name": "Olive Green", "RGB": (107, 142, 35)},
            {"Color Name": "Teal", "RGB": (0, 128, 128)},
            {"Color Name": "Pale Peach", "RGB": (255, 229, 180)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Burnt Orange", "RGB": (204, 85, 0)},
            {"Color Name": "Deep Burgundy", "RGB": (128, 0, 32)},
            {"Color Name": "Dusty Rose", "RGB": (205, 150, 158)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Brown",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Warm Grey", "RGB": (119, 136, 153)},
            {"Color Name": "Soft Pink", "RGB": (255, 182, 193)},
            {"Color Name": "Mint", "RGB": (189, 252, 201)},
            {"Color Name": "Cool Beige", "RGB": (210, 180, 140)},
            {"Color Name": "Steel Blue", "RGB": (70, 130, 180)},
            {"Color Name": "Soft Teal", "RGB": (96, 130, 182)},
            {"Color Name": "Dusty Blue", "RGB": (100, 149, 237)},
            {"Color Name": "Terracotta", "RGB": (204, 102, 51)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Brown",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Chocolate Brown", "RGB": (123, 63, 0)},
            {"Color Name": "Rose Gold", "RGB": (183, 110, 121)},
            {"Color Name": "Olive", "RGB": (128, 128, 0)},
            {"Color Name": "Dusty Blue", "RGB": (70, 130, 180)},
            {"Color Name": "Soft Violet", "RGB": (238, 130, 238)},
            {"Color Name": "Soft Coral", "RGB": (255, 127, 80)},
            {"Color Name": "Warm Taupe", "RGB": (210, 180, 140)},
            {"Color Name": "Light Periwinkle", "RGB": (197, 203, 255)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Blonde",
        "Eye Color": "Black",
        "Complementary Colors": [
            {"Color Name": "Warm Brown", "RGB": (139, 69, 19)},
            {"Color Name": "Champagne", "RGB": (250, 235, 215)},
            {"Color Name": "Coral", "RGB": (255, 127, 80)},
            {"Color Name": "Pale Green", "RGB": (152, 251, 152)},
            {"Color Name": "Soft Blue", "RGB": (135, 206, 235)},
            {"Color Name": "Soft Peach", "RGB": (255, 218, 185)},
            {"Color Name": "Taupe", "RGB": (72, 60, 50)},
            {"Color Name": "Coral", "RGB": (255, 127, 80)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Blonde",
        "Eye Color": "Brown",
        "Complementary Colors": [
            {"Color Name": "Beige", "RGB": (245, 245, 220)},
            {"Color Name": "Peach", "RGB": (255, 218, 185)},
            {"Color Name": "Soft Green", "RGB": (152, 251, 152)},
            {"Color Name": "Light Lavender", "RGB": (230, 230, 250)},
            {"Color Name": "Warm Pink", "RGB": (255, 182, 193)},
            {"Color Name": "Soft Mint", "RGB": (189, 252, 201)},
            {"Color Name": "Navy Blue", "RGB": (0, 0, 128)},
            {"Color Name": "Charcoal Gray", "RGB": (54, 69, 79)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Blonde",
        "Eye Color": "Gray",
        "Complementary Colors": [
            {"Color Name": "Pale Pink", "RGB": (250, 218, 221)},
            {"Color Name": "Mint Green", "RGB": (152, 255, 152)},
            {"Color Name": "Cream", "RGB": (255, 253, 208)},
            {"Color Name": "Soft Yellow", "RGB": (255, 255, 204)},
            {"Color Name": "Dusty Rose", "RGB": (205, 92, 92)},
            {"Color Name": "Cool Blue", "RGB": (173, 216, 230)},
            {"Color Name": "Mint Green", "RGB": (152, 255, 152)},
            {"Color Name": "Warm Taupe", "RGB": (210, 180, 140)},
        ],
    },
    {
        "Skin Undertone": "Neutral",
        "Hair Color": "Blonde",
        "Eye Color": "Blue",
        "Complementary Colors": [
            {"Color Name": "Taupe", "RGB": (72, 60, 50)},
            {"Color Name": "Cool Blue", "RGB": (173, 216, 230)},
            {"Color Name": "Warm Beige", "RGB": (245, 245, 220)},
            {"Color Name": "Light Coral", "RGB": (240, 128, 128)},
            {"Color Name": "Silver", "RGB": (192, 192, 192)},
            {"Color Name": "Light Periwinkle", "RGB": (197, 203, 255)},
            {"Color Name": "Coral Pink", "RGB": (255, 160, 122)},
            {"Color Name": "Lavender", "RGB": (230, 230, 250)},

        ],
    },
]

# Function to get recommended colors
def get_recommended_colors(undertone, hair_color, eye_color):
    for entry in dataset:
        if (entry["Skin Undertone"] == undertone and
            entry["Hair Color"] == hair_color and
            entry["Eye Color"] == eye_color):
            return entry["Complementary Colors"]
    return []

# Function to display color suggestions in the interface
def display_colors(colors):
    color_blocks = ""
    for color in colors:
        color_name = color["Color Name"]
        r, g, b = color["RGB"]
        color_block = f"<div style='display:inline-block; width:100px; height:100px; margin:10px; background-color:rgb({r},{g},{b});'></div><p>{color_name}</p>"
        color_blocks += color_block
    return color_blocks
    
# StreamLit interface function
def color_recommendation(image, hair_color, eye_color):
    if image is None:
        return "Please upload an image."

    image = Image.fromarray(image)
    undertone = find_dominant_skin_tone(image)

    recommended_colors = get_recommended_colors(undertone, hair_color, eye_color)
    if recommended_colors:
        return f"Skin Undertone: {undertone}<br>Hair Color: {hair_color}<br>Eye Color: {eye_color}<br><br>{display_colors(recommended_colors)}"
    else:
        return "No matching colors found in the dataset."

# Create StreamLit interface
iface = st.Interface(
    fn=color_recommendation,
    inputs=[
        st.Image(type="numpy", label="Upload Image"),
        st.Dropdown(choices=["Black", "Brown", "Blonde"], label="Select Hair Color"),
        st.Dropdown(choices=["Black", "Brown", "Blue", "Gray"], label="Select Eye Color")
    ],
    outputs=st.HTML(),
    title="Color Recommendation Based on Skin Undertone, Hair, and Eye Color",
    description="Upload an image of your face, select your hair and eye colors, and receive color recommendations."
)

# Launch the interface
iface.launch(debug=True)
