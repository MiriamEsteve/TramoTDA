from PIL import Image, ImageDraw, ImageFont

# Define customizable style options
TITLE_FONT_SIZE = 50
HEADER_FONT_SIZE = 30
TEXT_FONT_SIZE = 20
HEADER_COLOR = "darkblue"
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
IMAGE_SIZE = (1000, 500)  # Resize images for uniformity
PADDING = 20
MARGIN = 50
LINE_SPACING = 10
HEADER_HEIGHT = 60

# Load images and descriptions
steps = [
    ('1_Load_Trajectory_Data.png', 'Load Trajectory Data', 'Start by loading trajectory data, which could include spatial coordinates and timestamps from sources like GPS or AIS systems.'),
    ('2_Persistence_Diagrams.png', 'Generate Persistence Diagrams', 'Transform the trajectory data into persistence diagrams, capturing topological features that persist across various scales of the data. This step involves mathematical computations to identify significant structures within the data.'),
    ('3_Lifetime_Diagram.png', 'Generate Lifetime Diagrams', 'Calculate and plot the lifetime of each feature in the persistence diagrams.'),
    ('4_Persistence_Images.png', 'Generate Persistence Images', 'Visualize the persistence diagrams as images.'),
    ('5_Calculate_Barycenter.png', 'Calculate Barycenter', 'Compute the barycenter of the persistence diagrams to find a representative summary of the data sets. This step helps in reducing complexity and summarizing the topological features.'),
    ('6_Classification.png', 'Classification', 'Apply machine learning algorithms to classify the trajectories based on the features derived from the persistence diagrams and their barycenters. This might involve using classifiers like SVM, Random Forest, or neural networks, depending on the complexity and nature of the data.'),
    ('7_Evaluation_and_Refinement.png', 'Evaluation and Refinement', 'Assess the performance of the classification models using metrics such as accuracy, precision, and recall. Refine the models based on the outcomes to improve classification results.')
]

# Calculate the total height needed for the infographic
total_height = sum([IMAGE_SIZE[1] + HEADER_HEIGHT + 100 for _ in steps]) + 500  # additional space for descriptions and padding
flowchart_width = 1200
flowchart = Image.new('RGB', (flowchart_width, total_height), BACKGROUND_COLOR)
draw = ImageDraw.Draw(flowchart)

# Define fonts
try:
    title_font = ImageFont.truetype("arial.ttf", TITLE_FONT_SIZE)
    header_font = ImageFont.truetype("arial.ttf", HEADER_FONT_SIZE)
    text_font = ImageFont.truetype("arial.ttf", TEXT_FONT_SIZE)
except IOError:
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

# Define positions
y_offset = MARGIN
image_x_offset = (flowchart_width - IMAGE_SIZE[0]) // 2
text_x_offset = MARGIN

# Add title
title = "Trajectory Data Analysis and Classification"
title_bbox = draw.textbbox((0, 0), title, font=title_font)
draw.text(((flowchart_width - (title_bbox[2] - title_bbox[0])) // 2, y_offset), title, fill=HEADER_COLOR, font=title_font)
y_offset += title_bbox[3] - title_bbox[1] + 2 * PADDING

# Add steps with images and descriptions
colors = ["#FF5733", "#33C1FF", "#33FF57", "#FF33A1", "#FF8C33", "#B833FF", "#33FFDA"]

for (img_path, header, description), color in zip(steps, colors):
    # Draw header
    header_bbox = draw.textbbox((0, 0), header, font=header_font)
    draw.rectangle([(0, y_offset), (flowchart_width, y_offset + HEADER_HEIGHT)], fill=color)
    draw.text((text_x_offset, y_offset + (HEADER_HEIGHT - header_bbox[3]) // 2), header, fill=BACKGROUND_COLOR, font=header_font)

    y_offset += HEADER_HEIGHT + PADDING

    # Add image
    img = Image.open(img_path)
    img = img.resize(IMAGE_SIZE)
    flowchart.paste(img, (image_x_offset, y_offset))

    y_offset += img.height + PADDING

    # Add description
    description_lines = description.split('\n')
    for line in description_lines:
        text_bbox = draw.textbbox((text_x_offset, y_offset), line, font=text_font)
        draw.text((text_x_offset, y_offset), line, fill=TEXT_COLOR, font=text_font)
        y_offset += text_bbox[3] - text_bbox[1] + LINE_SPACING

    y_offset += 2 * PADDING

# Save the infographic image
flowchart.save('Infographic_Trajectory_Analysis.png')

# Display the infographic image
flowchart.show()
