# Temperature scheduling utilities for distillation
import math


class TemperatureScheduler:
    """Schedule temperature over training or generations.

    Higher temperature early transfers more "dark knowledge".
    Lower temperature later sharpens predictions.
    """

    def __init__(
        self,
        initial_temp: float = 20.0,
        final_temp: float = 1.0,
        schedule: str = "cosine",
        total_steps: int = 100,
    ):
        """Initialize temperature scheduler.

        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            schedule: Type of schedule ("constant", "linear", "cosine", "step")
            total_steps: Total steps/generations for scheduling
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule = schedule
        self.total_steps = total_steps

    def get_temperature(self, step: int) -> float:
        """Get temperature for current step/generation.

        Args:
            step: Current step or generation number

        Returns:
            Temperature value for this step
        """
        if self.schedule == "constant":
            return self.initial_temp

        progress = min(step / max(self.total_steps, 1), 1.0)

        if self.schedule == "linear":
            return self.initial_temp + (self.final_temp - self.initial_temp) * progress

        if self.schedule == "cosine":
            # Cosine annealing from initial to final
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.final_temp + (self.initial_temp - self.final_temp) * cosine_decay

        if self.schedule == "step":
            # Step decay at 1/3 and 2/3 progress
            if progress < 0.33:
                return self.initial_temp
            elif progress < 0.67:
                return (self.initial_temp + self.final_temp) / 2
            else:
                return self.final_temp

        return self.initial_temp

    def __repr__(self) -> str:
        return (
            f"TemperatureScheduler(initial={self.initial_temp}, "
            f"final={self.final_temp}, schedule={self.schedule})"
        )
