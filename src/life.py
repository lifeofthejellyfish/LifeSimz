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

# Constants and configuration
TICK_RATE = 60
TICK_INTERVAL = 1 / TICK_RATE
SIMULATION_SPEED = 0.25
global_sunlight = 0.1
TILE_SIZE = 32
DAY_NIGHT_CYCLE_SPEED = 0.5  # Speed of the day-night cycle

# Logging setup
LOG_FOLDER = "logs"
LOG_FILE = os.path.join(LOG_FOLDER, "application.log")
CONFIG_PATH = "config/config.json"
IMAGE_PATH = "assets/images"

# Global tracking data
tracking_data = {}
FOOD_WEB = {
    "normal": [],
    "carnivorous": ["bacteria", "fungus"],
    "herbivore": ["normal"],
    "carnivore": ["herbivore", "omnivore"],
    "omnivore": ["normal", "herbivore", "fungus"],
    "fungus": ["dead_matter"],
    "bacteria": ["organic_material"],
}

def invert_food_web(food_web):
    predator_map = {}
    for predator, prey_list in food_web.items():
        for prey in prey_list:
            predator_map.setdefault(prey, []).append(predator)
    return predator_map

PREDATOR_MAP = invert_food_web(FOOD_WEB)

def track_organisms(organisms, filename="organism_tracking.csv"):
    """Track the state of all organisms and save to a CSV file."""
    try:
        with open(filename, "a", newline="") as csvfile:
            fieldnames = ["time", "uuid", "type", "subtype", "x", "y", "energy", "age", "action"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                writer.writeheader()
            current_time = time.time()
            for organism in organisms:
                action = ""
                if organism.energy < 50:
                    action = "low_energy"
                if organism.energy == 0:
                    action = "died"
                writer.writerow({
                    "time": current_time,
                    "uuid": organism.uuid,
                    "type": organism.organism_type,
                    "subtype": organism.subtype,
                    "x": organism.x,
                    "y": organism.y,
                    "energy": organism.energy,
                    "age": organism.age,
                    "action": action,
                })
    except Exception as e:
        print(f"An error occurred during file operations: {e}")

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

pygame.init()

def load_config(config_file=CONFIG_PATH):
    try:
        with open(config_file, "r") as f:
            logging.info("Loading configuration from %s.", config_file)
            return json.load(f)
    except FileNotFoundError:
        logging.error("Configuration file '%s' not found.", config_file)
        sys.exit()
    except json.JSONDecodeError as e:
        logging.error("Error parsing '%s': %s", config_file, e)
        sys.exit()

config = load_config()

# Setup window dynamically
window_width = config["window"]["width"]
window_height = config["window"]["height"]
window_width = (window_width // TILE_SIZE) * TILE_SIZE
window_height = (window_height // TILE_SIZE) * TILE_SIZE

screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption(config["window"]["title"])

# Now that window dimensions are known, define GRID dimensions
GRID_WIDTH = window_width // TILE_SIZE
GRID_HEIGHT = window_height // TILE_SIZE

# Colors loaded from configuration
colors = config["colors"]
white = tuple(colors.get("white", [255, 255, 255]))
black = tuple(colors.get("black", [0, 0, 0]))

# Load sprite images from the updated path
sprite_images = {
    "bacteria": pygame.image.load(f"{IMAGE_PATH}/bacteria.png"),
    "carnivore": pygame.image.load(f"{IMAGE_PATH}/carnivore.png"),
    "carnivorous_plant": pygame.image.load(f"{IMAGE_PATH}/carnivorous_plant.png"),
    "fungus": pygame.image.load(f"{IMAGE_PATH}/fungus.png"),
    "herbivore": pygame.image.load(f"{IMAGE_PATH}/herbivore.png"),
    "omnivore": pygame.image.load(f"{IMAGE_PATH}/omnivore.png"),
    "plant": pygame.image.load(f"{IMAGE_PATH}/plant.png")
}

def assign_sprite(organism_type, subtype):
    """Assign sprite based on organism type and subtype."""
    if organism_type == "animal":
        if subtype == "carnivore":
            return sprite_images["carnivore"]
        elif subtype == "herbivore":
            return sprite_images["herbivore"]
        elif subtype == "omnivore":
            return sprite_images["omnivore"]
    elif organism_type == "plant":
        if subtype == "carnivorous":
            return sprite_images["carnivorous_plant"]
        else:
            return sprite_images["plant"]
    elif organism_type == "microorganism":
        if subtype == "fungus":
            return sprite_images["fungus"]
        elif subtype == "bacteria":
            return sprite_images["bacteria"]
    return None

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
        "metabolic_rate": random.uniform(genome_config["metabolic_rate"]["min"], genome_config["metabolic_rate"]["max"]),
        "body_mass": random.uniform(genome_config["body_mass"]["min"], genome_config["body_mass"]["max"]),
        "diet": random.choice(genome_config["diet"]),
        "red_gene": random.randint(0, 255),
        "green_gene": random.randint(0, 255),
        "blue_gene": random.randint(0, 255),
        "vision_radius": random.uniform(genome_config["vision_range"]["min"], genome_config["vision_range"]["max"]),
    }

def get_tile_at_position(x, y):
    grid_x = int(x // TILE_SIZE)
    grid_y = int(y // TILE_SIZE)
    if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
        return grid[grid_x][grid_y]
    logging.warning(f"Position ({x}, {y}) out of bounds. Returning default 'grass' tile.")
    return Tile(grid_x, grid_y, terrain_type="grass")

def get_neighboring_tiles(tile):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = tile.x + dx, tile.y + dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
            neighbors.append(grid[nx][ny])
    return neighbors

def draw_tooltip(surface, pos, organisms):
    closest = min(
        organisms,
        key=lambda o: math.hypot(o.x - pos[0], o.y - pos[1]),
        default=None
    )
    if closest and closest.rect.collidepoint(pos):
        font = pygame.font.Font(None, 24)
        tooltip = f"{closest.organism_type.capitalize()} ({closest.subtype.capitalize()}) - Age: {closest.age:.2f} yrs"
        text_surface = font.render(tooltip, True, black)
        tooltip_rect = text_surface.get_rect(topleft=(pos[0] + 10, pos[1] - 20))
        pygame.draw.rect(surface, white, tooltip_rect.inflate(6, 6))
        pygame.draw.rect(surface, black, tooltip_rect.inflate(6, 6), 1)
        surface.blit(text_surface, tooltip_rect.topleft)

def get_random_position():
    x = random.randint(0, GRID_WIDTH - 1) * TILE_SIZE + TILE_SIZE / 2
    y = random.randint(0, GRID_HEIGHT - 1) * TILE_SIZE + TILE_SIZE / 2
    return x, y

def draw_grid(surface):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            grid[x][y].draw(surface, TILE_SIZE)

class Tile:
    def __init__(self, x, y, terrain_type="grass"):
        self.x = x
        self.y = y
        self.terrain_type = terrain_type
        self.properties = self.get_properties_by_terrain()
        self.dead_matter = 0
        self.organic_material = 0
        self.has_plant = False

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
        self.uuid = str(uuid.uuid4())
        self.genome = genome
        self.organism_type = organism_type
        self.subtype = subtype
        self.x = x
        self.y = y
        self.size = size
        self.sprite = assign_sprite(organism_type, subtype)
        self.rect = self.sprite.get_rect(center=(self.x, self.y)) if self.sprite else None
        self.energy = 100 if "energy_rate" in genome else None
        self.last_update_time = time.time()
        self.age = 0
        self.body_mass = genome["body_mass"]
        self.max_longevity = genome["max_longevity"]
        self.adult_weight = genome["adult_weight"]
        self.metabolic_rate = genome["metabolic_rate"]
        self.diet = genome["diet"]
        self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.direction = self.normalize(self.direction)
        self.hunger_multiplier = self.get_energy_multiplier()
        logging.info(
            f"[CREATION] Organism {self.uuid} | Type: {self.organism_type}-{self.subtype} | "
            f"Energy: {self.energy} | Genome: {self.genome}"
        )
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
        return visible_organisms

    def sense_touch(self, grid):
        current_tile = get_tile_at_position(self.x, self.y)
        neighbors = get_neighboring_tiles(current_tile)
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
        current_tile = get_tile_at_position(self.x, self.y)
        neighbors = get_neighboring_tiles(current_tile)
        return {
            "current_tile": current_tile.terrain_type,
            "adjacent_terrain": [tile.terrain_type for tile in neighbors],
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
            resized_sprite = pygame.transform.scale(self.sprite, (int(self.size * 2), int(self.size * 2)))
            draw_x = self.x - self.size
            draw_y = self.y - self.size
            surface.blit(resized_sprite, (draw_x, draw_y))
            self.rect = resized_sprite.get_rect(center=(self.x, self.y))
        else:
            pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), int(self.size))
            self.rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size * 2, self.size * 2)

    def update(self, organisms, current_time):
        energy_decrement = 0  # Ensure energy_decrement is defined
        sensory_data = self.scan_environment(organisms, grid)
        self.age += time_step
        if self.age > self.max_longevity:
            logging.info(f"[DEATH] Organism {self.uuid} died of old age at age {self.age:.2f} years.")
            self.handle_death(organisms, grid)
            return
        if self.energy is not None and self.genome.get("metabolic_rate", 0) > 0:
            energy_decrement = self.genome["energy_rate"] * time_step * self.hunger_multiplier * 5
            self.energy = max(0, self.energy - energy_decrement)
        if self.organism_type == "plant" or (self.organism_type == "microorganism" and self.subtype == "fungus"):
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
            logging.info(
                f"[BEHAVIOR] {self.uuid} ({self.subtype}) detected predator {predator.uuid} ({predator.subtype}) and is fleeing."
            )
        elif prey_in_vision:
            prey = min(prey_in_vision, key=lambda o: math.hypot(o.x - self.x, o.y - self.y))
            dx = prey.x - self.x
            dy = prey.y - self.y
            self.direction = self.normalize([dx, dy])
            logging.info(
                f"[BEHAVIOR] {self.uuid} ({self.subtype}) detected prey {prey.uuid} ({prey.subtype}) and is moving towards it."
            )
        else:
            self.direction[0] += random.uniform(-0.1, 0.1)
            self.direction[1] += random.uniform(-0.1, 0.1)
            self.direction = self.normalize(self.direction)
        # Avoid water tiles
        ahead_x = self.x + self.direction[0] * self.genome.get("speed", 0) * time_step * 10
        ahead_y = self.y + self.direction[1] * self.genome.get("speed", 0) * time_step * 10

        ahead_tile = get_tile_at_position(ahead_x, ahead_y)
        if ahead_tile.terrain_type == "water" and self.organism_type != "microorganism":
            self.direction[0] += random.uniform(-0.5, 0.5)
            self.direction[1] += random.uniform(-0.5, 0.5)
            self.direction = self.normalize(self.direction)
        current_tile = get_tile_at_position(self.x, self.y)
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
            self.energy -= 40
            new_genome = self.genome.copy()
            for key in new_genome:
                if isinstance(new_genome[key], (int, float)):
                    mutation = random.uniform(-mutation_rate, mutation_rate)
                    new_genome[key] += mutation * new_genome[key]
            new_x = self.x + random.uniform(-TILE_SIZE, TILE_SIZE)
            new_y = self.y + random.uniform(-TILE_SIZE, TILE_SIZE)
            if self.organism_type == "animal":
                offspring = Animal(new_genome, new_x, new_y, self.size, self.subtype)
            elif self.organism_type == "plant":
                offspring = Plant(new_genome, new_x, new_y, self.size, self.subtype)
            elif self.organism_type == "microorganism":
                offspring = Microorganism(new_genome, new_x, new_y, self.size, self.subtype)
            organisms.append(offspring)
            logging.info(
                f"[REPRODUCTION] {self.uuid} created offspring {offspring.uuid} "
                f"at ({new_x:.2f}, {new_y:.2f}) with genome {offspring.genome}."
            )

    def handle_death(self, organisms, grid):
        current_tile = get_tile_at_position(self.x, self.y)
        if current_tile:
            dead_matter_added = self.body_mass * 0.1
            current_tile.add_dead_matter(dead_matter_added)
            logging.info(f"[DEATH] {self.uuid} added {dead_matter_added:.2f} dead matter to tile at ({current_tile.x}, {current_tile.y}).")
        if self in organisms:
            organisms.remove(self)

    def process_dead_matter(self, grid):
        if self.subtype == "fungus":
            current_tile = get_tile_at_position(self.x, self.y)
            if current_tile:
                decomposed = current_tile.decompose_dead_matter(2)
                self.energy += decomposed * 5
                if decomposed > 0:
                    logging.info(f"[FEED] {self.uuid} (fungus) decomposed {decomposed:.2f} dead matter at tile ({current_tile.x}, {current_tile.y}).")
                return decomposed > 0
        return False

    def feed(self, organisms, grid):
        if self.organism_type == "plant":
            return False
        prey_list = FOOD_WEB.get(self.subtype, [])
        for other in organisms[:]:
            if other.subtype in prey_list and other != self:
                distance = math.hypot(self.x - other.x, self.y - other.y)
                if distance <= self.proximity_radius:
                    self.energy += other.body_mass * 0.5
                    logging.info(f"[FEED] {self.uuid} ({self.subtype}) consumed {other.uuid} ({other.subtype}).")
                    organisms.remove(other)
                    return True
        if self.feed_on_tile_resources(grid):
            return True
        return False

    def feed_on_tile_resources(self, grid):
        current_tile = get_tile_at_position(self.x, self.y)
        if self.subtype == "herbivore" and current_tile.has_plant:
            self.energy += 20
            current_tile.has_plant = False
            logging.info(f"[FEED] {self.uuid} ({self.subtype}) consumed a plant at tile ({current_tile.x}, {current_tile.y}).")
            return True
        elif self.subtype == "fungus" and current_tile.dead_matter > 0:
            consumed = current_tile.decompose_dead_matter(2)
            self.energy += consumed * 5
            logging.info(f"[FEED] {self.uuid} ({self.subtype}) decomposed {consumed:.2f} dead matter at tile ({current_tile.x}, {current_tile.y}).")
            return True
        elif self.subtype == "bacteria" and current_tile.organic_material > 0:
            consumed = current_tile.consume_organic_material(1)
            self.energy += consumed * 10
            logging.info(f"[FEED] {self.uuid} ({self.subtype}) consumed {consumed:.2f} organic material at tile ({current_tile.x}, {current_tile.y}).")
            return True
        return False

class Animal(BaseOrganism):
    def __init__(self, genome, x, y, size, subtype):
        super().__init__(genome, x, y, size, "animal", subtype)
        self.energy = 120

class Plant(BaseOrganism):
    def __init__(self, genome, x, y, size, subtype):
        super().__init__(genome, x, y, size, "plant", subtype)
        self.genome["metabolic_rate"] = 0  # Plants have no metabolic rate
        self.last_reproduction_time = 0  # Track when the plant last reproduced
        current_tile = get_tile_at_position(self.x, self.y)
        current_tile.has_plant = True

    def handle_death(self, organisms, grid):
        super().handle_death(organisms, grid)
        current_tile = get_tile_at_position(self.x, self.y)
        if current_tile:
            current_tile.has_plant = False

    def update(self, organisms, current_time):
        super().update(organisms, current_time)
        global global_sunlight
        energy_replenishment = global_sunlight * self.size * 0.1
        self.energy = min(100, self.energy + energy_replenishment)

    def can_reproduce(self):
        reproduction_energy_threshold = 60
        required_age = self.genome["max_longevity"] / 5
        return self.energy > reproduction_energy_threshold and self.age >= required_age

    def reproduce(self, organisms, mutation_rate=0.05):
        current_time = time.time()
        reproduction_cooldown = 40  # Plants can reproduce only every 30 seconds

        if self.can_reproduce() and (current_time - self.last_reproduction_time > reproduction_cooldown):
            self.energy -= 51  # Increased energy cost for reproduction
            new_genome = self.genome.copy()

            # Apply mutations to offspring genome
            for key in new_genome:
                if isinstance(new_genome[key], (int, float)):
                    mutation = random.uniform(-mutation_rate, mutation_rate)
                    new_genome[key] += mutation * new_genome[key]

            # Produce a limited number of seeds
            num_offspring = random.randint(1, 2)  # Reduced from 3
            for _ in range(num_offspring):
                new_x = self.x + random.uniform(-2 * TILE_SIZE, 2 * TILE_SIZE)
                new_y = self.y + random.uniform(-2 * TILE_SIZE, 2 * TILE_SIZE)

                # Ensure offspring stays within bounds
                new_x = max(0, min(window_width, new_x))
                new_y = max(0, min(window_height, new_y))

                offspring = Plant(new_genome, new_x, new_y, self.size, self.subtype)
                organisms.append(offspring)

                logging.info(
                    f"[REPRODUCTION] {self.uuid} (plant) spread seed {offspring.uuid} "
                    f"at ({new_x:.2f}, {new_y:.2f}) with genome {offspring.genome}."
                )

            # Update the last reproduction time
            self.last_reproduction_time = current_time


class Microorganism(BaseOrganism):
    def __init__(self, genome, x, y, size, subtype):
        super().__init__(genome, x, y, size, "microorganism", subtype)

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
        screen.fill((255, 255, 255))
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

# Run main menu
num_creatures = main_menu()

# Improved terrain generation
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
WATER_PERCENT = 0.15
SAND_PERCENT = 0.10
GRASS_PERCENT = 0.60
FOREST_PERCENT = 0.15
sorted_heights = sorted([height for row in height_map for height in row])
num_tiles = GRID_WIDTH * GRID_HEIGHT
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

# Organism types
organism_types = {
    "animal": ["carnivore", "herbivore", "omnivore"],
    "microorganism": ["fungus", "bacteria"],
    "plant": ["normal", "carnivorous"]
}

# Generate organisms
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
    elif category == "plant":
        organisms.append(Plant(genome, x, y, size, subtype))

# Game loop
clock = pygame.time.Clock()
running = True
start_time = time.time()  # For day-night cycle
last_tick_time = time.time()
time_step = TICK_INTERVAL * SIMULATION_SPEED

while running:
    current_time = time.time()
    elapsed_time = current_time - start_time
    global_sunlight = (math.sin(elapsed_time * DAY_NIGHT_CYCLE_SPEED) + 1) / 2
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    while current_time - last_tick_time >= TICK_INTERVAL:
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
        # Track organisms' states at each step
        track_organisms(organisms)
        # Allow reproduction
        for organism in organisms[:]:
            organism.reproduce(organisms)
        last_tick_time += TICK_INTERVAL
    # Draw the grid
    draw_grid(screen)
    # Update and draw each organism
    for organism in organisms[:]:
        organism.update(organisms, current_time)
        organism.draw(screen)
    # Get the current mouse position
    mouse_pos = pygame.mouse.get_pos()
    # Draw the tooltip if hovering over an organism
    draw_tooltip(screen, mouse_pos, organisms)
    # Update the display
    pygame.display.flip()
