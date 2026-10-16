"""
Biologically inspired ant-colony food-search simulation.

Features
--------
- Correlated random exploration
- Local pheromone sensing
- Saturating pheromone response
- Recruitment pheromone laid by successful ants
- Pheromone evaporation
- Noisy path integration for homing
- Short-term spatial memory
- Learned food-location memory
- Individual variation between ants
- Local nest-search behaviour when homing is inaccurate
- Animated Matplotlib visualization
"""

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Deque, Dict, List, Optional, Tuple


Location = Tuple[int, int]
Direction = Tuple[int, int]




@dataclass
class FoodSource:
    """Food location and remaining quantity."""

    location: Location
    quantity: int = 100

    def take(self, amount: int = 1) -> int:
        """Remove food and return the amount actually collected."""

        collected = min(
            max(0, amount),
            self.quantity,
        )

        self.quantity -= collected

        return collected




class Ant:
    """
    A simplified biologically inspired foraging ant.

    Searching ants use:
    - directional persistence
    - local pheromone information
    - exploration
    - short-term memory
    - learned food-site memory

    Returning ants use:
    - noisy path integration
    - recruitment pheromone deposition
    - local nest searching if necessary
    """

    def __init__(
        self,
        colony: "AntColony",
        location: Optional[Location] = None,
    ) -> None:

        self.colony = colony

        self.location = (
            colony.nest_location
            if location is None
            else location
        )

        #state

        self.food_carried = 0
        self.returning_home = False

        self.direction: Optional[Direction] = None

        # Short-term memory only.
        # this is NOT used
        # to perfectly retrace the entire route.
        self.recent_locations: Deque[Location] = deque(
            maxlen=10
        )

        #path integration

        # Internal estimate of displacement from nest.
        self.home_vector_x = 0.0
        self.home_vector_y = 0.0

        #long term food memory
        self.known_food_location: Optional[Location] = None

        self.experience = 0

        
        # Individual variation
        

        self.pheromone_sensitivity = random.uniform(
            0.8,
            1.2,
        )

        self.navigation_noise = random.uniform(
            0.06,
            0.16,
        )

        self.persistence = random.uniform(
            0.85,
            1.15,
        )

        # When an ant believes it is near home but cannot
        # locate the nest, it performs a local search.
        self.searching_for_nest = False


   
    # FOOD DETECTION
   

    def search_for_food(self) -> bool:
        """Pick up food when standing on the food source."""

        if self.returning_home:
            return False

        food = self.colony.food_source

        if (
            self.location == food.location
            and food.quantity > 0
        ):

            collected = food.take(1)

            if collected > 0:

                self.food_carried = collected
                self.returning_home = True

                # rewarding location.
                self.known_food_location = food.location

                # Strong chemical signal near the food.
                self.colony.add_food_pheromone(
                    self.location,
                    self.colony.food_discovery_pheromone,
                )

                return True

        return False


    
    # OUTBOUND EXPLORATION
    

    def explore(self) -> None:
        """
        Search for food.

        Movement is probabilistic rather than deterministic.
        """

        neighbors = self._neighbors()

        if not neighbors:
            return

        weights = []

        for destination in neighbors:

            weight = 1.0

            dx = destination[0] - self.location[0]
            dy = destination[1] - self.location[1]

            
            # Directional persistence
            

            if self.direction is not None:

                old_dx, old_dy = self.direction

                dot_product = (
                    dx * old_dx
                    + dy * old_dy
                )

                if dot_product > 0:

                    weight *= (
                        self.colony.persistence_strength
                        * self.persistence
                    )

                elif dot_product < 0:

                    # Discourage immediately reversing.
                    weight *= 0.45

            
            # Recruitment pheromone
            

            pheromone = (
                self.colony.sensed_food_pheromone(
                    destination
                )
            )

            # Saturating response.
            
            # Extremely high pheromone levels therefore do not
            # make every ant deterministically choose the same
            # location.
            pheromone_response = (
                pheromone
                / (
                    pheromone
                    + self.colony.pheromone_half_saturation
                )
            )

            weight *= (
                1.0
                + self.colony.pheromone_influence
                * self.pheromone_sensitivity
                * pheromone_response
            )

            
            # Private food memory
            

            if self.known_food_location is not None:

                if (
                    random.random()
                    < self.colony.memory_use_probability
                ):

                    current_distance = self._distance(
                        self.location,
                        self.known_food_location,
                    )

                    new_distance = self._distance(
                        destination,
                        self.known_food_location,
                    )

                    if new_distance < current_distance:

                        weight *= (
                            1.0
                            + self.colony.memory_influence
                        )

            
            # Short-term memory
            

            if destination in self.recent_locations:

                weight *= 0.20

            
            # Exploration noise
            

            weight *= random.uniform(
                0.70,
                1.30,
            )

            weights.append(
                max(weight, 0.001)
            )

        destination = random.choices(
            neighbors,
            weights=weights,
            k=1,
        )[0]

        self._move(destination)


    
    # RETURN TO NEST
    

    def return_to_nest(self) -> None:
        """
        Return toward the nest using noisy path integration.

        The ant does NOT retrace a stored path.
        """

        if not self.returning_home:
            return

        # Deposit recruitment pheromone.
        self._deposit_recruitment_pheromone()

    
        # Reached nest
    

        if self.location == self.colony.nest_location:

            self._deliver_food()

            return

    
        # Determine estimated nest position
    

        estimated_nest = self._estimated_nest_location()

        actual_distance = self._distance(
            self.location,
            self.colony.nest_location,
        )

        # If very close to the real nest, increase local
        # searching behaviour.
        if (
            actual_distance
            <= self.colony.nest_search_radius
        ):

            self.searching_for_nest = True

        if self.searching_for_nest:

            self._local_nest_search()

        else:

            self._home_using_path_integration(
                estimated_nest
            )


    def _home_using_path_integration(
        self,
        estimated_nest: Tuple[float, float],
    ) -> None:
        """Move according to the internally estimated nest position."""

        neighbors = self._neighbors()

        if not neighbors:
            return

        weights = []

        for destination in neighbors:

            distance = self._distance(
                destination,
                estimated_nest,
            )

            # Prefer movements reducing estimated distance.
            weight = math.exp(
                -self.colony.homing_strength
                * distance
            )

            
            # Nest/home chemical information
            

            home_signal = (
                self.colony.sensed_home_pheromone(
                    destination
                )
            )

            response = (
                home_signal
                / (
                    home_signal
                    + self.colony.pheromone_half_saturation
                )
            )

            weight *= (
                1.0
                + self.colony.home_pheromone_influence
                * response
            )

            
            # Direction persistence


            dx = destination[0] - self.location[0]
            dy = destination[1] - self.location[1]

            if self.direction == (dx, dy):

                weight *= 1.4

            # Navigation noise.
            weight *= random.uniform(
                0.70,
                1.30,
            )

            weights.append(
                max(weight, 0.0001)
            )

        destination = random.choices(
            neighbors,
            weights=weights,
            k=1,
        )[0]

        self._move(destination)


    
    # LOCAL NEST SEARCH
    

    def _local_nest_search(self) -> None:
        """
        Search locally for the nest.

        This creates the looping/searching behaviour that occurs
        when path integration brings an ant close to home but not
        exactly to the entrance.
        """

        neighbors = self._neighbors()

        if not neighbors:
            return

        weights = []

        for destination in neighbors:

            home_signal = (
                self.colony.sensed_home_pheromone(
                    destination
                )
            )

            actual_distance = self._distance(
                destination,
                self.colony.nest_location,
            )

            # Local nest cues become important.
            weight = (
                1.0
                + home_signal * 2.0
            )

            # Slight geometric preference prevents ants from
            # wandering indefinitely around the nest.
            weight *= math.exp(
                -0.35 * actual_distance
            )

            if destination in self.recent_locations:
                weight *= 0.35

            weight *= random.uniform(
                0.75,
                1.25,
            )

            weights.append(
                max(weight, 0.001)
            )

        destination = random.choices(
            neighbors,
            weights=weights,
            k=1,
        )[0]

        self._move(destination)


    
    # PATH INTEGRATION
    

    def _estimated_nest_location(
        self,
    ) -> Tuple[float, float]:
        """
        Estimate the nest position using the ant's internal
        home vector.

        Noise increases with travel distance.
        """

        estimated_x = (
            self.location[0]
            - self.home_vector_x
        )

        estimated_y = (
            self.location[1]
            - self.home_vector_y
        )

        travelled_distance = math.hypot(
            self.home_vector_x,
            self.home_vector_y,
        )

        noise_scale = (
            self.navigation_noise
            * math.sqrt(
                max(travelled_distance, 1.0)
            )
        )

        estimated_x += random.gauss(
            0,
            noise_scale,
        )

        estimated_y += random.gauss(
            0,
            noise_scale,
        )

        return (
            estimated_x,
            estimated_y,
        )


    
    # PHEROMONE DEPOSITION
    

    def _deposit_recruitment_pheromone(
        self,
    ) -> None:
        """
        Deposit recruitment pheromone while carrying food.

        Deposition decreases on already saturated trails.
        """

        existing = (
            self.colony.food_pheromones.get(
                self.location,
                0.0,
            )
        )

        saturation_factor = (
            1.0
            / (
                1.0
                + existing
                / self.colony.deposit_saturation
            )
        )

        # Experienced successful ants can produce slightly
        # stronger recruitment.
        experience_bonus = min(
            1.0 + self.experience * 0.03,
            1.4,
        )

        amount = (
            self.colony.return_pheromone_amount
            * saturation_factor
            * experience_bonus
        )

        self.colony.add_food_pheromone(
            self.location,
            amount,
        )


    
    # DELIVER FOOD
    

    def _deliver_food(self) -> None:

        self.colony.food_collected += (
            self.food_carried
        )

        self.food_carried = 0

        self.returning_home = False

        self.experience += 1

        self.searching_for_nest = False

        # Reset path integrator.
        self.home_vector_x = 0.0
        self.home_vector_y = 0.0

        self.direction = None

        self.recent_locations.clear()


    
    # MOVEMENT
    

    def _move(
        self,
        destination: Location,
    ) -> None:

        old_x, old_y = self.location

        new_x, new_y = destination

        dx = new_x - old_x
        dy = new_y - old_y

        self.recent_locations.append(
            self.location
        )

        self.direction = (
            dx,
            dy,
        )

        self.location = destination

        
        # Update path integrator
        

        integration_error_x = random.gauss(
            0,
            self.navigation_noise * 0.05,
        )

        integration_error_y = random.gauss(
            0,
            self.navigation_noise * 0.05,
        )

        self.home_vector_x += (
            dx + integration_error_x
        )

        self.home_vector_y += (
            dy + integration_error_y
        )


    
    # NEIGHBORS
    

    def _neighbors(self) -> List[Location]:

        x, y = self.location

        size = self.colony.world_size

        possible = [

            (x - 1, y),
            (x + 1, y),

            (x, y - 1),
            (x, y + 1),

            (x - 1, y - 1),
            (x - 1, y + 1),

            (x + 1, y - 1),
            (x + 1, y + 1),
        ]

        return [

            (nx, ny)

            for nx, ny in possible

            if (
                0 <= nx < size
                and 0 <= ny < size
            )
        ]


    @staticmethod
    def _distance(
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )



# COLONY


class AntColony:
    """Manage ants, food, pheromones, and simulation state."""

    def __init__(
        self,
        num_ants: int,
        food_source: FoodSource,
        world_size: int = 100,
        nest_location: Location = (0, 0),
    ) -> None:

       
        # Validation
        

        if num_ants < 0:

            raise ValueError(
                "num_ants must be non-negative"
            )

        if world_size < 1:

            raise ValueError(
                "world_size must be positive"
            )

        for name, location in (

            ("nest_location", nest_location),

            (
                "food source",
                food_source.location,
            ),

        ):

            if not all(
                0 <= coordinate < world_size
                for coordinate in location
            ):

                raise ValueError(
                    f"{name} must be inside "
                    f"the world bounds"
                )

        
        # World
        

        self.world_size = world_size

        self.nest_location = nest_location

        self.food_source = food_source

        
        self.initial_food_quantity = (
            food_source.quantity
        )

        
        # Chemical fields
        

        self.food_pheromones: Dict[
            Location,
            float,
        ] = {}

        self.home_pheromones: Dict[
            Location,
            float,
        ] = {}

        
        # Statistics
        

        self.food_collected = 0
        self.step_count = 0

        
        # Behaviour parameters
        

        self.persistence_strength = 2.3

        self.pheromone_influence = 5.0

        self.home_pheromone_influence = 1.5

        # Controls pheromone saturation.
        self.pheromone_half_saturation = 3.0

        # Probability of using private food memory.
        self.memory_use_probability = 0.35

        self.memory_influence = 2.8

        
        self.homing_strength = 0.42

        
        
        self.nest_search_radius = 4.0

        # Recruitment trail parameters.
        self.return_pheromone_amount = 2.0

        self.food_discovery_pheromone = 6.0

        self.deposit_saturation = 12.0

        # Chemical decay.
        self.food_pheromone_evaporation = 0.035

        self.home_pheromone_evaporation = 0.008

        
        # Ant population
        

        self.colony: List[Ant] = [

            Ant(self)

            for _ in range(num_ants)
        ]

        # Establish weak nest odor.
        self._initialize_nest_odor()


    
    # NEST ODOR
    

    def _initialize_nest_odor(self) -> None:
        """
        Creating a small chemical region surrounding the nest.

        This represents short-range nest cues rather than
        a world-wide navigation beacon.
        """

        nx, ny = self.nest_location

        radius = 5

        for x in range(
            max(0, nx - radius),
            min(
                self.world_size,
                nx + radius + 1,
            ),
        ):

            for y in range(
                max(0, ny - radius),
                min(
                    self.world_size,
                    ny + radius + 1,
                ),
            ):

                distance = math.hypot(
                    x - nx,
                    y - ny,
                )

                if distance <= radius:

                    strength = (
                        5.0
                        * (
                            1.0
                            - distance
                            / (radius + 1)
                        )
                    )

                    current = (
                        self.home_pheromones.get(
                            (x, y),
                            0.0,
                        )
                    )

                    self.home_pheromones[
                        (x, y)
                    ] = max(
                        current,
                        strength,
                    )


    
    # PHEROMONES
    

    def add_food_pheromone(
        self,
        location: Location,
        amount: float,
    ) -> None:

        self.food_pheromones[location] = (

            self.food_pheromones.get(
                location,
                0.0,
            )

            + amount
        )


    def sensed_food_pheromone(
        self,
        location: Location,
    ) -> float:

        return self._sense_field(
            self.food_pheromones,
            location,
        )


    def sensed_home_pheromone(
        self,
        location: Location,
    ) -> float:

        return self._sense_field(
            self.home_pheromones,
            location,
        )


    def _sense_field(
        self,
        field: Dict[Location, float],
        location: Location,
    ) -> float:
        """
        Approximate local chemical sensing.

        Instead of reading one exact square, the ant samples
        the immediate surrounding area.
        """

        x, y = location

        total = 0.0

        for dx in (-1, 0, 1):

            for dy in (-1, 0, 1):

                px = x + dx
                py = y + dy

                if not (
                    0 <= px < self.world_size
                    and 0 <= py < self.world_size
                ):
                    continue

                strength = field.get(
                    (px, py),
                    0.0,
                )

                distance = math.hypot(
                    dx,
                    dy,
                )

                if distance == 0:

                    contribution = strength

                else:

                    contribution = (
                        strength
                        / (1.0 + distance)
                    )

                total += contribution

        # Sensory measurement is imperfect.
        total *= random.uniform(
            0.85,
            1.15,
        )

        return total


    
    # SIMULATION STEP
    

    def step(self) -> None:

        # Randomize activation order
        # This prevents ant #0 from always changing the
        # environment before ant #199.
        ants = list(self.colony)

        random.shuffle(ants)

        for ant in ants:

            if ant.returning_home:

                ant.return_to_nest()

            else:

                found_food = (
                    ant.search_for_food()
                )

                if not found_food:

                    ant.explore()

                    ant.search_for_food()

        self.evaporate_pheromones()

        self.step_count += 1


    
    # EVAPORATION
    

    def evaporate_pheromones(self) -> None:

        self._evaporate_field(
            self.food_pheromones,
            self.food_pheromone_evaporation,
        )

        self._evaporate_field(
            self.home_pheromones,
            self.home_pheromone_evaporation,
        )

        # Nest odor is continually renewed.
        self._initialize_nest_odor()


    @staticmethod
    def _evaporate_field(
        field: Dict[Location, float],
        rate: float,
    ) -> None:

        for location in list(field):

            field[location] *= (
                1.0 - rate
            )

            if field[location] < 0.02:

                del field[location]


    
    # NON-GRAPHICAL SIMULATION
    

    def simulate(
        self,
        steps: int,
    ) -> None:

        if steps < 0:

            raise ValueError(
                "steps must be non-negative"
            )

        for _ in range(steps):

            self.step()


    
    # ANIMATION
    

    def visualize(
        self,
        steps: int = 3000,
        interval: int = 20,
    ) -> None:
        """
        Animate the colony.

        Colors:
        - white     = searching ants
        - green     = ants carrying food
        - blue      = nest
        - red       = food
        - orange    = recruitment pheromone
        """

        import matplotlib.pyplot as plt

        from matplotlib.animation import (
            FuncAnimation,
        )

        import numpy as np

        
        # Figure
        

        figure, axis = plt.subplots(
            figsize=(10, 9)
        )

        figure.patch.set_facecolor(
            "#101216"
        )

        axis.set_facecolor(
            "#181b20"
        )

        axis.set_xlim(
            -0.5,
            self.world_size - 0.5,
        )

        axis.set_ylim(
            -0.5,
            self.world_size - 0.5,
        )

        axis.set_aspect("equal")

        axis.set_xlabel(
            "X",
            color="white",
        )

        axis.set_ylabel(
            "Y",
            color="white",
        )

        axis.tick_params(
            colors="#aaaaaa"
        )

        for spine in axis.spines.values():

            spine.set_color(
                "#555555"
            )

        
        # Recruitment pheromone heat map
        

        pheromone_grid = np.zeros(
            (
                self.world_size,
                self.world_size,
            )
        )

        pheromone_image = axis.imshow(

            pheromone_grid,

            origin="lower",

            extent=(

                -0.5,

                self.world_size - 0.5,

                -0.5,

                self.world_size - 0.5,

            ),

            cmap="inferno",

            vmin=0,

            vmax=8,

            alpha=0.70,

            interpolation="bilinear",

            zorder=1,
        )

        
        # Searching ants
        

        searching_scatter = axis.scatter(

            [],

            [],

            s=14,

            c="#eeeeee",

            alpha=0.75,

            linewidths=0,

            label="Searching ants",

            zorder=5,
        )

        
        # Returning ants
        

        returning_scatter = axis.scatter(

            [],

            [],

            s=26,

            c="#39d98a",

            alpha=0.95,

            edgecolors="white",

            linewidths=0.3,

            label="Carrying food",

            zorder=6,
        )

        
        # Nest-searching ants
        

        nest_search_scatter = axis.scatter(

            [],

            [],

            s=28,

            facecolors="none",

            edgecolors="#54b9ff",

            linewidths=1.0,

            label="Searching for nest",

            zorder=7,
        )

        
        # Nest
        

        axis.scatter(

            *self.nest_location,

            marker="h",

            s=280,

            c="#4da6ff",

            edgecolors="white",

            linewidths=1.5,

            label="Nest",

            zorder=10,
        )

        
        # Food
        

        food_marker = axis.scatter(

            *self.food_source.location,

            marker="*",

            s=350,

            c="#ff4d5a",

            edgecolors="white",

            linewidths=1.0,

            label="Food",

            zorder=10,
        )

       
        # Legend
       

        legend = axis.legend(

            loc="upper right",

            facecolor="#22252d",

            edgecolor="#555555",

            fontsize=9,
        )

        for text in legend.get_texts():

            text.set_color(
                "white"
            )

        
        # Title
        

        title = axis.set_title(

            "Biologically Inspired Ant Foraging",

            color="white",

            fontsize=15,

            pad=12,
        )

        
        # Information box
        

        info_text = axis.text(

            0.01,

            0.99,

            "",

            transform=axis.transAxes,

            verticalalignment="top",

            color="white",

            fontsize=10,

            bbox={

                "boxstyle": "round",

                "facecolor": "#111318",

                "alpha": 0.85,

                "edgecolor": "#555555",

            },

            zorder=20,
        )

        
        # Update animation
        

        def update(frame: int):

            self.step()

            
            # Pheromone grid
            

            grid = np.zeros(
                (
                    self.world_size,
                    self.world_size,
                )
            )

            for (
                x,
                y,
            ), strength in (
                self.food_pheromones.items()
            ):

                grid[y, x] = strength

            pheromone_image.set_data(
                grid
            )

            max_strength = max(
                self.food_pheromones.values(),
                default=1.0,
            )

            pheromone_image.set_clim(

                0,

                max(
                    4.0,
                    min(
                        max_strength,
                        20.0,
                    ),
                ),
            )

            
            # Separate ant states
            

            searching = [

                ant.location

                for ant in self.colony

                if not ant.returning_home
            ]

            returning = [

                ant.location

                for ant in self.colony

                if (
                    ant.returning_home
                    and not ant.searching_for_nest
                )
            ]

            nest_searching = [

                ant.location

                for ant in self.colony

                if (
                    ant.returning_home
                    and ant.searching_for_nest
                )
            ]

            
            # Update scatter plots
            

            if searching:

                searching_scatter.set_offsets(
                    searching
                )

            else:

                searching_scatter.set_offsets(
                    np.empty((0, 2))
                )

            if returning:

                returning_scatter.set_offsets(
                    returning
                )

            else:

                returning_scatter.set_offsets(
                    np.empty((0, 2))
                )

            if nest_searching:

                nest_search_scatter.set_offsets(
                    nest_searching
                )

            else:

                nest_search_scatter.set_offsets(
                    np.empty((0, 2))
                )

            
            # Food marker shrinks as food disappears
            

            if (
                self.initial_food_quantity > 0
            ):

                food_fraction = (

                    self.food_source.quantity

                    / self.initial_food_quantity
                )

            else:

                food_fraction = 0

            if self.food_source.quantity > 0:

                food_size = (
                    80
                    + 270
                    * food_fraction
                )

            else:

                food_size = 25

            food_marker.set_sizes(
                [food_size]
            )

            
            # Statistics
            

            experienced = sum(

                ant.experience > 0

                for ant in self.colony
            )

            average_experience = (

                sum(
                    ant.experience
                    for ant in self.colony
                )

                / max(
                    len(self.colony),
                    1,
                )
            )

            info_text.set_text(

                f"Step: {self.step_count}\n"

                f"Food remaining: "
                f"{self.food_source.quantity}\n"

                f"Food delivered: "
                f"{self.food_collected}\n"

                f"Searching ants: "
                f"{len(searching)}\n"

                f"Returning ants: "
                f"{len(returning)}\n"

                f"Nest-searching ants: "
                f"{len(nest_searching)}\n"

                f"Experienced ants: "
                f"{experienced}\n"

                f"Mean trips/ant: "
                f"{average_experience:.2f}\n"

                f"Pheromone cells: "
                f"{len(self.food_pheromones)}"
            )

            
            # Title
            

            if self.food_source.quantity > 0:

                title.set_text(
                    "Ant Colony — "
                    "Foraging & Recruitment"
                )

            else:

                title.set_text(
                    "Ant Colony — "
                    "Food Source Depleted"
                )

            return (

                pheromone_image,

                searching_scatter,

                returning_scatter,

                nest_search_scatter,

                food_marker,

                info_text,

                title,
            )

        
        # Animation
        

        animation = FuncAnimation(

            figure,

            update,

            frames=steps,

            interval=interval,

            blit=False,

            repeat=False,

            cache_frame_data=False,
        )

        # Keep reference alive.
        self._animation = animation

        plt.tight_layout()

        plt.show()



# RUN PROGRAM


if __name__ == "__main__":

    
    random.seed(7)

    colony = AntColony(

        num_ants=200,

        food_source=FoodSource(

            location=(40, 40),

            quantity=150,
        ),

        world_size=60,

        nest_location=(8, 8),
    )

    
    # START ANIMATED SIMULATION
    

    colony.visualize(

        steps=3000,

        interval=20,
    )

    

    print(
        f"Food remaining: "
        f"{colony.food_source.quantity}"
    )

    print(
        f"Food delivered to nest: "
        f"{colony.food_collected}"
    )
