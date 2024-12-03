import os
import sys
import time
import uuid
import json
import math
import random
import logging
import pygame
import pygame_gui
import csv
import copy
from deap import base, creator, tools, algorithms

# ---------------------------
# Constants and Configuration
# ---------------------------

# Default configuration
DEFAULT_CONFIG = {
    "window": {"width": 800, "height": 600, "title": "Creature Simulation"},
    "colors": {
        "white": [255, 255, 255],
        "black": [0, 0, 0],
    },
    "genome": {
        "growth_rate": {"min": 0.1, "max": 1.0},
        "max_size": {"min": 10, "max": 50},
        "speed": {"min": 1.0, "max": 5.0},
        "metabolic_rate": {"min": 0.1, "max": 1.0},
        "spore_rate": {"min": 0.0, "max": 0.5},
        "reproduction_rate": {"min": 0.1, "max": 1.0},
        "predation_rate": {"min": 0.1, "max": 1.0},
        "trap_rate": {"min": 0.0, "max": 0.3},
        "size": {"min": 5.0, "max": 25.0},
        "adult_weight": {"min": 1.0, "max": 10.0},
        "max_longevity": {"min": 5.0, "max": 20.0},
        "body_mass": {"min": 0.5, "max": 5.0},
        "diet": ["herbivore", "carnivore", "omnivore"],
        "vision_range": {"min": 50, "max": 200},
    },
    "energy_multipliers": {
        "animal": {"carnivore": 1.2, "herbivore": 1.0, "omnivore": 1.1},
        "microorganism": {"fungus": 1.3, "bacteria": 1.0},
    }
}

# Paths
LOG_FOLDER = "logs"
LOG_FILE = os.path.join(LOG_FOLDER, "application.log")
CONFIG_PATH = "config/config.json"
IMAGE_PATH = "assets/images"

# Simulation Constants
TILE_SIZE = 40
DAY_NIGHT_CYCLE_SPEED = 0.5  # Speed of the day-night cycle
SIMULATION_SPEED = 0.25
global_sunlight = 0.1

# Initialize global tracking data
tracking_data = {}
FOOD_WEB = {
    "herbivore": [],
    "carnivore": ["herbivore", "omnivore"],
    "omnivore": ["herbivore", "fungus"],
    "fungus": ["dead_matter"],
    "bacteria": ["organic_material"],
}

# Genetic Algorithm Constants
GA_POPULATION_SIZE = 20
GA_NUM_GENERATIONS = 10
GA_CXPB = 0.5  # Crossover probability
GA_MUTPB = 0.2  # Mutation probability
GA_NUM_PARAMETERS = 5  # DAY_NIGHT_CYCLE_SPEED, global_sunlight, WATER_PERCENT, SAND_PERCENT

# ---------------------------
# Setup Logging
# ---------------------------

def setup_logging():
    """Set up logging to log folder and console."""
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)
    logging.info("Logging setup complete. Logs will be written to '%s'.", LOG_FILE)

# Call setup_logging before anything else
setup_logging()
logging.info("Starting the program...")

# ---------------------------
# Load Configuration
# ---------------------------

def load_config(config_file=CONFIG_PATH):
    try:
        with open(config_file, "r") as f:
            logging.info("Loading configuration from %s.", config_file)
            return json.load(f)
    except FileNotFoundError:
        logging.warning("Configuration file '%s' not found. Using default configuration.", config_file)
        return copy.deepcopy(DEFAULT_CONFIG)
    except json.JSONDecodeError as e:
        logging.error("Error parsing '%s': %s. Using default configuration.", config_file, e)
        return copy.deepcopy(DEFAULT_CONFIG)

config = load_config()

# ---------------------------
# Initialize Pygame
# ---------------------------

pygame.init()

# Setup window dynamically
window_width = config["window"]["width"]
window_height = config["window"]["height"]
window_width = (window_width // TILE_SIZE) * TILE_SIZE
window_height = (window_height // TILE_SIZE) * TILE_SIZE

screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption(config["window"]["title"])

# Define GRID dimensions
GRID_WIDTH = window_width // TILE_SIZE
GRID_HEIGHT = window_height // TILE_SIZE

# Colors loaded from configuration
colors = config["colors"]
white = tuple(colors.get("white", [255, 255, 255]))
black = tuple(colors.get("black", [0, 0, 0]))

# ---------------------------
# Load Sprite Images
# ---------------------------

def load_sprite_images():
    sprite_keys = ["bacteria", "carnivore", "fungus", "herbivore", "omnivore"]
    sprite_images = {}
    for key in sprite_keys:
        sprite_path = os.path.join(IMAGE_PATH, f"{key}.png")
        if os.path.exists(sprite_path):
            try:
                sprite_images[key] = pygame.image.load(sprite_path).convert_alpha()
                logging.info("Loaded sprite image for '%s'.", key)
            except pygame.error as e:
                logging.error("Error loading sprite '%s.png': %s. Using default circle.", key, e)
                sprite_images[key] = None
        else:
            logging.warning("Sprite image '%s.png' not found in '%s'. Using default circle.", key, IMAGE_PATH)
            sprite_images[key] = None
    return sprite_images

sprite_images = load_sprite_images()

# ---------------------------
# Define Classes
# ---------------------------

class Tile:
    def __init__(self, x, y, terrain_type="grass"):
        self.x = x
        self.y = y
        self.terrain_type = terrain_type
        self.properties = self.get_properties_by_terrain()
        self.dead_matter = 0
        self.organic_material = 0

    def get_properties_by_terrain(self):
        terrain_properties = {
            "grass": {"movement_cost": 1, "color": (124, 252, 0)},
            "sand": {"movement_cost": 1.5, "color": (237, 201, 175)},
            "water": {"movement_cost": 2, "color": (64, 164, 223)},
            "forest": {"movement_cost": 1.2,"color": (34, 139, 34)},
        }
        return terrain_properties.get(self.terrain_type, {"movement_cost": 1, "color": (0, 0, 0)})

    def draw(self, surface, tile_size):
        color = self.properties["color"]
        rect = pygame.Rect(self.x * tile_size, self.y * tile_size, tile_size, tile_size)
        pygame.draw.rect(surface, color, rect)

    def add_dead_matter(self, amount):
        self.dead_matter += amount

    def decompose_dead_matter(self, amount):
        decomposed = min(self.dead_matter, amount)
        self.dead_matter -= decomposed
        self.organic_material += decomposed
        return decomposed

    def consume_organic_material(self, amount):
        consumed = min(self.organic_material, amount)
        self.organic_material -= consumed
        return consumed

class BaseOrganism:
    def __init__(self, genome, x, y, size, organism_type, subtype=None):
        self.visible_uuids = []
        self.action = "born"
        self.uuid = str(uuid.uuid4())
        self.genome = genome
        self.organism_type = organism_type
        self.subtype = subtype
        self.x = x
        self.y = y
        self.size = size
        self.sprite = assign_sprite(organism_type, subtype)
        self.rect = self.sprite.get_rect(center=(self.x, self.y)) if self.sprite else None
        self.last_update_time = time.time()
        self.age = 0
        self.body_mass = genome["body_mass"]
        self.max_longevity = genome["max_longevity"]
        self.adult_weight = genome["adult_weight"]
        self.metabolic_rate = genome["metabolic_rate"]
        self.diet = genome["diet"]
        self.energy = genome["max_size"] * 5
        self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.direction = self.normalize(self.direction)
        self.hunger_multiplier = self.get_energy_multiplier()
        self.vision_radius = genome.get("vision_radius", 7 * TILE_SIZE)
        self.proximity_radius = genome.get("proximity_radius", int(3 * TILE_SIZE))
        self.hearing_range = genome.get("hearing_range", 7 * TILE_SIZE)
        self.smell_range = genome.get("smell_range", 10 * TILE_SIZE)

    def scan_environment(self, organisms, grid):
        sensory_data = {
            "vision": self.sense_vision(organisms),
            "touch": self.sense_touch(grid),
            "proximity": self.sense_proximity(organisms),
            "energy": self.sense_energy(),
            "environment": self.sense_environment(grid),
            "direction": self.sense_direction(),
        }
        return sensory_data

    def sense_vision(self, organisms):
        visible_organisms = [
            org for org in organisms
            if math.hypot(org.x - self.x, org.y - self.y) <= self.vision_radius and org != self
        ]
        self.visible_uuids = [org.uuid for org in visible_organisms]
        return visible_organisms

    def sense_touch(self, grid):
        current_tile = get_tile_at_position(self.x, self.y, grid)
        neighbors = get_neighboring_tiles(current_tile, grid)
        return {"current_tile": current_tile, "adjacent_tiles": neighbors}

    def sense_proximity(self, organisms):
        nearby_organisms = [
            org for org in organisms
            if math.hypot(org.x - self.x, org.y - self.y) <= self.proximity_radius and org != self
        ]
        return nearby_organisms

    def sense_energy(self):
        return {
            "energy": self.energy,
            "age": self.age,
            "hunger": self.energy <= self.genome["energy_rate"] * 5,
        }

    def sense_environment(self, grid):
        current_tile = get_tile_at_position(self.x, self.y, grid)
        neighbors = get_neighboring_tiles(current_tile, grid)
        return {
            "current_tile": current_tile.terrain_type if current_tile else "unknown",
            "adjacent_terrain": [tile.terrain_type for tile in neighbors] if neighbors else [],
        }

    def sense_direction(self):
        return {
            "near_boundary": (
                self.x <= self.size or
                self.x >= window_width - self.size or
                self.y <= self.size or
                self.y >= window_height - self.size
            ),
            "direction": self.direction,
        }

    def normalize(self, vector):
        length = math.hypot(vector[0], vector[1])
        if length == 0:
            return [random.uniform(-1, 1), random.uniform(-1, 1)]
        return [vector[0] / length, vector[1] / length]

    def get_energy_multiplier(self):
        multipliers = config.get("energy_multipliers", {})
        type_multipliers = multipliers.get(self.organism_type, {})
        return type_multipliers.get(self.subtype, 1.0)

    def draw(self, surface):
        if self.vision_radius:
            pygame.draw.circle(
                surface,
                (0, 0, 255),
                (int(self.x), int(self.y)),
                int(self.vision_radius),
                1
            )
        if self.proximity_radius:
            pygame.draw.circle(
                surface,
                (255, 0, 0),
                (int(self.x), int(self.y)),
                int(self.proximity_radius),
                1
            )
        if self.sprite:
            try:
                resized_sprite = pygame.transform.scale(self.sprite, (int(self.size * 2), int(self.size * 2)))
                draw_x = self.x - self.size
                draw_y = self.y - self.size
                surface.blit(resized_sprite, (draw_x, draw_y))
                self.rect = resized_sprite.get_rect(center=(self.x, self.y))
            except Exception as e:
                logging.error("Error drawing sprite for organism %s: %s", self.uuid, e)
                pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(self.size))
                self.rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)
        else:
            pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(self.size))
            self.rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

    def update(self, organisms, current_time, grid, time_step):
        energy_decrement = 0
        sensory_data = self.scan_environment(organisms, grid)
        self.age += time_step

        if self.age > self.max_longevity:
            self.handle_death(organisms, grid)
            self.action = "died_of_old_age"
            return

        if self.energy is not None and self.genome.get("metabolic_rate", 0) > 0:
            energy_decrement += self.genome["metabolic_rate"] * time_step * self.hunger_multiplier * 5
            self.energy = max(0, self.energy - energy_decrement)

        if self.organism_type == "microorganism" and self.subtype == "fungus":
            self.last_update_time = current_time
            return

        # Adjust movement direction based on predator avoidance and prey detection
        prey_list = FOOD_WEB.get(self.subtype, [])
        predator_list = PREDATOR_MAP.get(self.subtype, [])
        prey_in_vision = [
            org for org in sensory_data['vision']
            if org.subtype in prey_list and org != self
        ]
        predators_in_vision = [
            org for org in sensory_data['vision']
            if org.subtype in predator_list and org != self
        ]

        if predators_in_vision:
            predator = min(predators_in_vision, key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
            dx = self.x - predator.x
            dy = self.y - predator.y
            self.direction = self.normalize([dx, dy])

        elif prey_in_vision:
            prey = min(prey_in_vision, key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
            dx = prey.x - self.x
            dy = prey.y - self.y
            self.direction = self.normalize([dx, dy])

        else:
            self.direction[0] += random.uniform(-0.1, 0.1)
            self.direction[1] += random.uniform(-0.1, 0.1)
            self.direction = self.normalize(self.direction)

        # Avoid water tiles
        ahead_x = self.x + self.direction[0] * self.genome.get("speed", 0) * time_step * 10
        ahead_y = self.y + self.direction[1] * self.genome.get("speed", 0) * time_step * 10

        ahead_tile = get_tile_at_position(ahead_x, ahead_y, grid)
        if ahead_tile and ahead_tile.terrain_type == "water" and self.organism_type != "microorganism":
            self.direction[0] += random.uniform(-0.5, 0.5)
            self.direction[1] += random.uniform(-0.5, 0.5)
            self.direction = self.normalize(self.direction)

        current_tile = get_tile_at_position(self.x, self.y, grid)
        movement_cost = current_tile.properties["movement_cost"] if current_tile else 1
        adjusted_speed = max(0.5, self.genome.get("speed", 0) / movement_cost) * 2
        dx = self.direction[0] * adjusted_speed * time_step * 10
        dy = self.direction[1] * adjusted_speed * time_step * 10
        self.x += dx
        self.y += dy

        # Constrain to world bounds
        self.x = max(self.size, min(window_width - self.size, self.x))
        self.y = max(self.size, min(window_height - self.size, self.y))

        # Update energy with movement cost
        energy_decrement += (math.hypot(dx, dy) * self.body_mass * 0.1)
        self.energy = max(0, self.energy - energy_decrement)
        self.last_update_time = current_time

    def can_reproduce(self):
        reproduction_energy_threshold = 80
        required_age = self.genome["max_longevity"] / 3
        return self.energy > reproduction_energy_threshold and self.age >= required_age

    def reproduce(self, organisms, mutation_rate=0.05):
        if self.can_reproduce():
            offspring_energy = self.energy * 0.5  # New offspring gets 50% of parent's current energy
            self.energy *= 0.5  # Parent retains the other half
            new_genome = self.genome.copy()

            # Apply mutations
            for key in new_genome:
                if isinstance(new_genome[key], (int, float)):
                    mutation = random.uniform(-mutation_rate, mutation_rate)
                    new_genome[key] += mutation * new_genome[key]

            new_x = self.x + random.uniform(-TILE_SIZE, TILE_SIZE)
            new_y = self.y + random.uniform(-TILE_SIZE, TILE_SIZE)

            if self.organism_type == "animal":
                offspring = Animal(new_genome, new_x, new_y, self.size, self.subtype, energy=offspring_energy)
            elif self.organism_type == "microorganism":
                offspring = Microorganism(new_genome, new_x, new_y, self.size, self.subtype, energy=offspring_energy)
            self.action = "reproduced"

            organisms.append(offspring)
            logging.info("Organism %s reproduced. Offspring UUID: %s", self.uuid, offspring.uuid)

    def handle_death(self, organisms, grid):
        current_tile = get_tile_at_position(self.x, self.y, grid)
        if current_tile and self.subtype != "fungus":
            current_tile.organic_material += self.body_mass * 0.1  # Add some organic material
            logging.debug(f"Organism {self.uuid} died. Added organic material to tile ({current_tile.x}, {current_tile.y}).")
        if self in organisms:
            organisms.remove(self)
            logging.info(f"Organism {self.uuid} removed from simulation.")

    def process_dead_matter(self, grid):
        if self.subtype == "fungus":
            current_tile = get_tile_at_position(self.x, self.y, grid)
            max_consume_rate = 5  # Fungi can consume at most 5 units of dead matter per tick
            if current_tile and current_tile.dead_matter > 0:
                decomposed = current_tile.decompose_dead_matter(max_consume_rate)
                self.energy += decomposed * 3  # Gain energy for decomposed matter
                logging.debug(f"Fungus {self.uuid} decomposed {decomposed} dead matter.")
                return True
        return False

    def feed(self, organisms, grid):
        prey_list = FOOD_WEB.get(self.subtype, [])
        for other in organisms[:]:
            if other.subtype in prey_list and other != self:
                distance = math.hypot(self.x - other.x, self.y - other.y)
                if distance <= self.proximity_radius:
                    self.energy += other.body_mass * 0.5
                    self.action = "Ate"
                    logging.info("Organism %s ate organism %s.", self.uuid, other.uuid)
                    organisms.remove(other)
                    return True
        if self.feed_on_tile_resources(grid):
            return True
        return False

    def feed_on_tile_resources(self, grid):
        current_tile = get_tile_at_position(self.x, self.y, grid)
        if self.subtype == "bacteria" and current_tile and current_tile.organic_material > 0:
            consumed = current_tile.consume_organic_material(1)
            self.energy += consumed * 10
            self.action = "bacteria fed"
            logging.info("Bacteria %s fed on tile organic material.", self.uuid)
            return True
        return False

class Animal(BaseOrganism):
    def __init__(self, genome, x, y, size, subtype, energy=100):
        super().__init__(genome, x, y, size, "animal", subtype)
        self.energy = energy

class Microorganism(BaseOrganism):
    def __init__(self, genome, x, y, size, subtype, energy=100):
        super().__init__(genome, x, y, size, "microorganism", subtype)
        self.last_reproduction_time = None  # Keep track of the last reproduction time
        self.energy = energy

    def can_reproduce(self):
        reproduction_cooldown = 10  # Fungus can reproduce once every 10 seconds
        required_age = self.genome["max_longevity"] / 5  # Must reach 20% of lifespan
        min_energy_for_reproduction = 80  # Require at least 80 energy to reproduce

        if (
            self.age < required_age or
            self.energy < min_energy_for_reproduction or
            (self.last_reproduction_time and time.time() - self.last_reproduction_time < reproduction_cooldown)
        ):
            return False
        return True

    def reproduce(self, organisms, mutation_rate=0.05):
        if self.subtype == "fungus" and self.can_reproduce():
            self.last_reproduction_time = time.time()
            offspring_count = random.randint(1, 2)  # 1 or 2 offspring
            for _ in range(offspring_count):
                new_genome = copy.deepcopy(self.genome)
                for key in new_genome:
                    if isinstance(new_genome[key], (int, float)):
                        mutation = random.uniform(-mutation_rate, mutation_rate)
                        new_genome[key] += mutation * new_genome[key]

                new_x = self.x + random.uniform(-2 * TILE_SIZE, 2 * TILE_SIZE)
                new_y = self.y + random.uniform(-2 * TILE_SIZE, 2 * TILE_SIZE)

                offspring = Microorganism(new_genome, new_x, new_y, self.size, "fungus", energy=self.energy * 0.3)
                self.energy *= 0.7  # Transfer 30% of energy to each offspring
                organisms.append(offspring)
                logging.info("Fungus %s reproduced. Offspring UUID: %s", self.uuid, offspring.uuid)

# ---------------------------
# Utility Functions
# ---------------------------

def invert_food_web(food_web):
    predator_map = {}
    for predator, prey_list in food_web.items():
        for prey in prey_list:
            predator_map.setdefault(prey, []).append(predator)
    return predator_map

PREDATOR_MAP = invert_food_web(FOOD_WEB)

def assign_sprite(organism_type, subtype):
    """Assign sprite based on organism type and subtype."""
    sprite_key = None
    if organism_type == "animal":
        if subtype == "carnivore":
            sprite_key = "carnivore"
        elif subtype == "herbivore":
            sprite_key = "herbivore"
        elif subtype == "omnivore":
            sprite_key = "omnivore"
    elif organism_type == "microorganism":
        if subtype == "fungus":
            sprite_key = "fungus"
        elif subtype == "bacteria":
            sprite_key = "bacteria"

    if sprite_key and sprite_images.get(sprite_key):
        return sprite_images[sprite_key]
    return None

def track_organisms(organisms, filename="organism_tracking.csv"):
    """Track the state of all organisms and save to a CSV file."""
    try:
        with open(filename, "a", newline="") as csvfile:
            fieldnames = ["time", "uuid", "type", "subtype", "x", "y", "energy", "age", "action", "sees","genome"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                writer.writeheader()
            current_time = time.time()
            for organism in organisms:
                action = ""
                if organism.energy < 50:
                    action = "low_energy"
                if organism.energy == 0:
                    action = "died of energy loss"
                if organism.action == "Ate":
                    action = "Ate"
                if organism.action == "died_of_old_age":
                    action = "died_of_old_age"
                if organism.action == "reproduced":
                    action = "reproduced"
                if organism.action == "born":
                    action = "born"
                if organism.action == "dies":
                    action = "Died"
                if organism.action == "bacteria fed":
                    action = "bacteria fed"
                if organism.action == "fungus decomposed stuff":
                    action = "fungus decomposed stuff"
                writer.writerow({
                    "time": current_time,
                    "uuid": organism.uuid,
                    "type": organism.organism_type,
                    "subtype": organism.subtype,
                    "x": organism.x,
                    "y": organism.y,
                    "sees": ", ".join(organism.visible_uuids) if organism.visible_uuids else "None",
                    "energy": organism.energy,
                    "age": organism.age,
                    "action": action,
                    "genome": json.dumps(organism.genome)
                })
    except Exception as e:
        logging.error(f"An error occurred during file operations: {e}")

def draw_tooltip(surface, pos, organisms, grid):
    if not organisms:
        return  # No organisms to display

    # Determine the closest organism
    closest = None
    min_distance = float('inf')
    for organism in organisms:
        distance = math.hypot(organism.x - pos[0], organism.y - pos[1])
        if distance < min_distance:
            min_distance = distance
            closest = organism

    # Get the tile at the mouse position
    grid_x = int(pos[0] // TILE_SIZE)
    grid_y = int(pos[1] // TILE_SIZE)
    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
        tile = grid[grid_x][grid_y]
    else:
        tile = None

    # Tooltip content
    tooltip_lines = []

    # Add organism information to tooltip if hovering over an organism
    if closest and closest.rect and closest.rect.collidepoint(pos):
        tooltip_lines.append(f"{closest.organism_type.capitalize()} ({closest.subtype.capitalize()})")
        tooltip_lines.append(f"Age: {closest.age:.2f} yrs")
        tooltip_lines.append(f"Energy: {closest.energy:.2f}")

    # Add tile information to tooltip
    if tile:
        tooltip_lines.append(f"Tile Dead Matter: {tile.dead_matter:.2f}")
        tooltip_lines.append(f"Tile Organic Material: {tile.organic_material:.2f}")

    # Draw the tooltip if there’s content
    if tooltip_lines:
        font = pygame.font.Font(None, 24)
        line_height = 20
        tooltip_width = max(font.size(line)[0] for line in tooltip_lines) + 10
        tooltip_height = len(tooltip_lines) * line_height + 5
        tooltip_x = pos[0] + 10
        tooltip_y = pos[1] - tooltip_height

        # Ensure tooltip stays within window bounds
        if tooltip_x + tooltip_width > window_width:
            tooltip_x = pos[0] - tooltip_width - 10
        if tooltip_y < 0:
            tooltip_y = pos[1] + 10

        # Draw tooltip background
        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(surface, (255, 255, 255), tooltip_rect)  # White background
        pygame.draw.rect(surface, (0, 0, 0), tooltip_rect, 1)    # Black border

        # Draw each line of text
        for i, line in enumerate(tooltip_lines):
            text_surface = font.render(line, True, (0, 0, 0))
            surface.blit(text_surface, (tooltip_x + 5, tooltip_y + 5 + i * line_height))

def get_random_position():
    x = random.randint(0, GRID_WIDTH - 1) * TILE_SIZE + TILE_SIZE / 2
    y = random.randint(0, GRID_HEIGHT - 1) * TILE_SIZE + TILE_SIZE / 2
    return x, y

def draw_grid(surface, grid):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            grid[x][y].draw(surface, TILE_SIZE)

def get_tile_at_position(x, y, grid):
    grid_x = int(x // TILE_SIZE)
    grid_y = int(y // TILE_SIZE)
    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
        return grid[grid_x][grid_y]
    logging.warning(f"Position ({x}, {y}) out of bounds. Returning default 'grass' tile.")
    return Tile(grid_x, grid_y, terrain_type="grass")

def get_neighboring_tiles(tile, grid):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx = tile.x + dx
        ny = tile.y + dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
            neighbors.append(grid[nx][ny])
    return neighbors

def generate_genome(genome_config):
    """Generate a comprehensive genome with all potential traits."""
    return {
        "growth_rate": random.uniform(genome_config["growth_rate"]["min"], genome_config["growth_rate"]["max"]),
        "max_size": random.uniform(genome_config["max_size"]["min"], genome_config["max_size"]["max"]),
        "speed": max(0.5, random.uniform(genome_config["speed"]["min"], genome_config["speed"]["max"])),
        "energy_rate": random.uniform(genome_config["metabolic_rate"]["min"], genome_config["metabolic_rate"]["max"]),
        "spore_rate": random.uniform(genome_config["spore_rate"]["min"], genome_config["spore_rate"]["max"]),
        "reproduction_rate": random.uniform(genome_config["reproduction_rate"]["min"], genome_config["reproduction_rate"]["max"]),
        "predation_rate": random.uniform(genome_config["predation_rate"]["min"], genome_config["predation_rate"]["max"]),
        "trap_rate": random.uniform(genome_config["trap_rate"]["min"], genome_config["trap_rate"]["max"]),
        "size": random.uniform(genome_config["size"]["min"], genome_config["size"]["max"]),
        "adult_weight": random.uniform(genome_config["adult_weight"]["min"], genome_config["adult_weight"]["max"]),
        "max_longevity": random.uniform(genome_config["max_longevity"]["min"], genome_config["max_longevity"]["max"]),
        "metabolic_rate": max(0.2, min(0.8, random.uniform(genome_config["metabolic_rate"]["min"], genome_config["metabolic_rate"]["max"]))),
        "body_mass": random.uniform(genome_config["body_mass"]["min"], genome_config["body_mass"]["max"]),
        "diet": random.choice(genome_config["diet"]),
        "red_gene": random.randint(0, 255),
        "green_gene": random.randint(0, 255),
        "blue_gene": random.randint(0, 255),
        "vision_radius": random.uniform(genome_config["vision_range"]["min"], genome_config["vision_range"]["max"] * (0.8 if genome_config["diet"] == "carnivore" else 1.2)),
    }

def initialize_organisms(num_creatures, config, grid):
    organisms = []
    for _ in range(num_creatures):
        category = random.choice(list(organism_types.keys()))
        subtype = random.choice(organism_types[category])
        genome = generate_genome(config["genome"])
        x, y = get_random_position()
        size = random.uniform(10, 20)
        if category == "animal":
            organisms.append(Animal(genome, x, y, size, subtype))
        elif category == "microorganism":
            organisms.append(Microorganism(genome, x, y, size, subtype))
    return organisms

def initialize_grid(WATER_PERCENT, SAND_PERCENT, GRASS_PERCENT, FOREST_PERCENT):
    height_map = [[random.uniform(0, 10) for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
    NUM_SMOOTHING_ITERATIONS = 5
    for _ in range(NUM_SMOOTHING_ITERATIONS):
        new_height_map = [[0 for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                sum_heights = 0
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                            sum_heights += height_map[nx][ny]
                            count += 1
                new_height_map[x][y] = sum_heights / count
        height_map = copy.deepcopy(new_height_map)
    sorted_heights = sorted([height for row in height_map for height in row])
    num_tiles = GRID_WIDTH * GRID_HEIGHT
    # Clamp WATER_PERCENT and SAND_PERCENT to ensure W + S <= 0.25
    if WATER_PERCENT + SAND_PERCENT > 0.25:
        SAND_PERCENT = max(0.0, 0.25 - WATER_PERCENT)
        logging.warning("Adjusted SAND_PERCENT to %.2f to maintain W + S <= 0.25.", SAND_PERCENT)
    WATER_THRESHOLD = sorted_heights[int(WATER_PERCENT * num_tiles)]
    SAND_THRESHOLD = sorted_heights[int((WATER_PERCENT + SAND_PERCENT) * num_tiles)]
    GRASS_THRESHOLD = sorted_heights[int((WATER_PERCENT + SAND_PERCENT + GRASS_PERCENT) * num_tiles)]
    grid = [[None for _ in range(GRID_HEIGHT)] for _ in range(GRID_WIDTH)]
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            h = height_map[x][y]
            if h <= WATER_THRESHOLD:
                terrain_type = "water"
            elif h <= SAND_THRESHOLD:
                terrain_type = "sand"
            elif h <= GRASS_THRESHOLD:
                terrain_type = "grass"
            else:
                terrain_type = "forest"
            grid[x][y] = Tile(x, y, terrain_type)
    return grid

# ---------------------------
# Genetic Algorithm Setup
# ---------------------------

# Define the fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Register attributes for each parameter with appropriate ranges
toolbox.register("attr_sim_speed", random.uniform, 0.1, 2.0)
toolbox.register("attr_day_night_speed", random.uniform, 0.1, 2.0)
toolbox.register("attr_global_sunlight", random.uniform, 0.0, 1.0)
toolbox.register("attr_water_percent", random.uniform, 0.0, 0.25)  # Ensuring W <= 0.25
toolbox.register("attr_sand_percent", random.uniform, 0.0, 0.25)   # Ensuring S <= 0.25

# Define individual as a combination of the above attributes
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_sim_speed, toolbox.attr_day_night_speed,
                  toolbox.attr_global_sunlight, toolbox.attr_water_percent,
                  toolbox.attr_sand_percent), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def run_simulation_with_params(params, config, num_creatures):
    # Unpack parameters
    DAY_NIGHT_CYCLE_SPEED_param, global_sunlight_param, WATER_PERCENT, SAND_PERCENT = params

    # Fixed terrain percentages
    GRASS_PERCENT = 0.60
    FOREST_PERCENT = 0.15

    # Initialize grid based on parameters
    grid = initialize_grid(WATER_PERCENT, SAND_PERCENT, GRASS_PERCENT, FOREST_PERCENT)

    # Initialize organisms
    organisms = initialize_organisms(num_creatures, config, grid)

    # Run the simulation for a set number of ticks
    num_ticks = 2000  # Example number of ticks
    start_time_sim = time.time()
    last_tick_time_sim = start_time_sim
    time_step_sim = (1 / 60) * SIMULATION_SPEED  # Assuming TICK_RATE=60

    for _ in range(num_ticks):
        current_time_sim = time.time()
        elapsed_time_sim = current_time_sim - start_time_sim
        current_sunlight = (math.sin(elapsed_time_sim * DAY_NIGHT_CYCLE_SPEED_param) + 1) / 2 * global_sunlight_param

        # Simulate ticks
        for organism in organisms[:]:
            organism.feed(organisms, grid)
            if organism.energy <= 0:
                organism.handle_death(organisms, grid)

        for organism in organisms[:]:
            organism.reproduce(organisms)

        for organism in organisms[:]:
            organism.update(organisms, current_time_sim, grid, time_step_sim)

    # Collect metrics
    surviving_species = len(set(org.subtype for org in organisms))
    population_fluctuation = sum(abs(len([o for o in organisms if o.subtype == s]) - 10) for s in FOOD_WEB.keys())
    remaining_resources = sum(tile.dead_matter + tile.organic_material for row in grid for tile in row)

    return {
        "surviving_species": surviving_species,
        "population_fluctuation": population_fluctuation,
        "remaining_resources": remaining_resources
    }

def calculate_fitness(simulation_data):
    # Extract metrics
    surviving_species = simulation_data["surviving_species"]
    population_fluctuation = simulation_data["population_fluctuation"]
    remaining_resources = simulation_data["remaining_resources"]

    # Fitness function
    fitness = (
        10 * surviving_species - 5 * population_fluctuation - 0.1 * remaining_resources
    )
    return fitness

def evaluate(individual):
    # Map the individual's parameters to the simulation
    DAY_NIGHT_CYCLE_SPEED_param, global_sunlight_param, WATER_PERCENT, SAND_PERCENT = individual

    # Run the simulation and collect metrics
    simulation_data = run_simulation_with_params(individual, config, num_creatures_ga)

    # Calculate and return fitness
    return calculate_fitness(simulation_data),

# Register operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Placeholder for number of creatures in GA
num_creatures_ga = 10  # This will be set before running GA

def run_ga(config, num_creatures):
    global num_creatures_ga
    num_creatures_ga = num_creatures

    # Create the initial population
    population = toolbox.population(n=GA_POPULATION_SIZE)
    num_generations = GA_NUM_GENERATIONS

    logging.info("Starting Genetic Algorithm with population size %d and %d generations.", GA_POPULATION_SIZE, num_generations)
    logging.info("Initial population: %s", population)

    # Run the Genetic Algorithm
    algorithms.eaSimple(
        population, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB, ngen=num_generations, verbose=True
    )

    # Select the best individual from the final population
    best_individual = tools.selBest(population, k=1)[0]
    logging.info("Best Parameters: %s", best_individual)
    print("Best Parameters:", best_individual)
    return best_individual

def apply_best_params(params):
    global SIMULATION_SPEED, DAY_NIGHT_CYCLE_SPEED, global_sunlight
    SIMULATION_SPEED, DAY_NIGHT_CYCLE_SPEED, global_sunlight = params[:3]
    # WATER_PERCENT and SAND_PERCENT are used during terrain initialization
    WATER_PERCENT, SAND_PERCENT = params[3], params[4]
    logging.info(f"Applied Parameters: SIMULATION_SPEED={SIMULATION_SPEED}, "
                 f"DAY_NIGHT_CYCLE_SPEED={DAY_NIGHT_CYCLE_SPEED}, global_sunlight={global_sunlight}, "
                 f"WATER_PERCENT={WATER_PERCENT}, SAND_PERCENT={SAND_PERCENT}")
    print(f"Applied Parameters: SIMULATION_SPEED={SIMULATION_SPEED}, "
          f"DAY_NIGHT_CYCLE_SPEED={DAY_NIGHT_CYCLE_SPEED}, global_sunlight={global_sunlight}, "
          f"WATER_PERCENT={WATER_PERCENT}, SAND_PERCENT={SAND_PERCENT}")

# ---------------------------
# User Interface (Main Menu)
# ---------------------------

def main_menu():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Creature Simulation Menu")
    manager = pygame_gui.UIManager((window_width, window_height))
    title_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((window_width // 2 - 150, 50), (300, 50)),
        text="Creature Simulation Menu",
        manager=manager,
        object_id="#title_label"
    )
    start_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((window_width // 2 - 100, 150), (200, 50)),
        text="Start Simulation",
        manager=manager
    )
    quit_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((window_width // 2 - 100, 220), (200, 50)),
        text="Quit",
        manager=manager
    )
    slider_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((window_width // 2 - 150, 300), (300, 50)),
        text="Number of Creatures: 10",
        manager=manager
    )
    slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((window_width // 2 - 150, 350), (300, 30)),
        start_value=10,
        value_range=(1, 50),
        manager=manager
    )
    running = True
    num_creatures = 10
    while running:
        time_delta = clock.tick(60) / 1000.0
        screen.fill(white)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == start_button:
                    running = False
                elif event.ui_element == quit_button:
                    pygame.quit()
                    sys.exit()
            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == slider:
                    num_creatures = int(slider.get_current_value())
                    slider_label.set_text(f"Number of Creatures: {num_creatures}")
            manager.process_events(event)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()
    return num_creatures

# ---------------------------
# Simulation Classes and Functions
# ---------------------------

# Organism types
organism_types = {
    "animal": ["carnivore", "herbivore", "omnivore"],
    "microorganism": ["fungus", "bacteria"]
}

def initialize_organisms_ga(num_creatures, config, grid):
    return initialize_organisms(num_creatures, config, grid)

# ---------------------------
# Main Simulation Loop
# ---------------------------

def run_simulation():
    # Run main menu to get number of creatures
    num_creatures = main_menu()

    # Optionally run GA before starting the simulation
    apply_ga = True  # Set to False if you don't want to run GA every time
    if apply_ga:
        best_params = run_ga(config, num_creatures)
        apply_best_params(best_params)
    else:
        # Use default or config parameters
        best_params = [SIMULATION_SPEED, DAY_NIGHT_CYCLE_SPEED, global_sunlight, 0.15, 0.10]
        apply_best_params(best_params)

    # Extract GA parameters or use defaults
    if apply_ga:
        DAY_NIGHT_CYCLE_SPEED_param, global_sunlight_param, WATER_PERCENT, SAND_PERCENT = best_params
    else:
        DAY_NIGHT_CYCLE_SPEED_param, global_sunlight_param, WATER_PERCENT, SAND_PERCENT = SIMULATION_SPEED, DAY_NIGHT_CYCLE_SPEED, global_sunlight, 0.15, 0.10

    # Initialize grid
    grid = initialize_grid(WATER_PERCENT, SAND_PERCENT, GRASS_PERCENT=0.60, FOREST_PERCENT=0.15)

    # Initialize organisms
    organisms = initialize_organisms(num_creatures, config, grid)

    # Game loop variables
    clock = pygame.time.Clock()
    running = True
    start_time_sim = time.time()
    last_tick_time_sim = start_time_sim
    time_step_sim = (1 / 60) * SIMULATION_SPEED  # Assuming TICK_RATE=60

    while running:
        # Check if all organisms are dead
        if not organisms:
            print("All organisms have died. Ending simulation.")
            logging.info("All organisms have died. Simulation ending.")
            running = False
            break

        current_time_sim = time.time()
        elapsed_time_sim = current_time_sim - start_time_sim
        global_sunlight_current = (math.sin(elapsed_time_sim * DAY_NIGHT_CYCLE_SPEED_param) + 1) / 2 * global_sunlight_param

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        while current_time_sim - last_tick_time_sim >= (1 / 60):
            # Allow organisms to feed before checking for death
            for organism in organisms[:]:
                if organism.energy < 50:
                    organism.feed(organisms, grid)

            # Now check for death due to energy depletion
            for organism in organisms[:]:
                if organism.energy <= 0:
                    organism.handle_death(organisms, grid)

            # Process dead matter for fungi
            for organism in organisms[:]:
                if organism.subtype == "fungus":
                    organism.process_dead_matter(grid)

            # Bacteria feed on organic material
            for organism in organisms[:]:
                if organism.subtype == "bacteria":
                    organism.feed(organisms, grid)

            # Allow reproduction
            for organism in organisms[:]:
                organism.reproduce(organisms)

            # Track organisms' states at each step
            track_organisms(organisms)

            last_tick_time_sim += (1 / 60)

        # Draw the grid
        draw_grid(screen, grid)

        # Update and draw each organism
        for organism in organisms[:]:
            organism.update(organisms, current_time_sim, grid, time_step_sim)
            organism.draw(screen)

        # Get the current mouse position
        mouse_pos = pygame.mouse.get_pos()

        # Draw the tooltip if hovering over an organism
        draw_tooltip(screen, mouse_pos, organisms, grid)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    run_simulation()
