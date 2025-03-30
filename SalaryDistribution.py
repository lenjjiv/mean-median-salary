import math
from scipy.stats import norm

class SalaryDistribution:
    """
    Класс для моделирования распределения зарплат через логнормальное распределение.
    """
    
    def __init__(self, median, mean, population_size=None):
        """
        Инициализирует распределение с заданными параметрами
        
        Args:
            median: Медианная зарплата
            mean: Средняя зарплата
            population_size: Размер популяции (опционально)
        """
        self.median = median
        self.mean = mean
        self.population_size = population_size
        
        # Вычисляем параметры логнормального распределения
        self.mu, self.sigma = self._get_lognormal_params()
    
    def _get_lognormal_params(self):
        """
        Вычисляет параметры логнормального распределения
        
        Returns:
            tuple: (mu, sigma) параметры логнормального распределения
        """
        mu = math.log(self.median)
        sigma = math.sqrt(2 * math.log(self.mean / self.median))
        return mu, sigma
    
    def quantile_to_salary(self, quantile):
        """
        Возвращает зарплату по заданному квантилю
        
        Args:
            quantile: Значение квантиля (от 0 до 1)
        
        Returns:
            float: Зарплата, соответствующая квантилю
        """
        z = norm.ppf(quantile)
        threshold = math.exp(self.mu + self.sigma * z)
        return threshold
    
    def salary_to_quantile(self, salary):
        """
        Возвращает квантиль по заданной зарплате
        
        Args:
            salary: Значение зарплаты
        
        Returns:
            float: Квантиль (от 0 до 1)
        """
        z = (math.log(salary) - self.mu) / self.sigma
        quantile = norm.cdf(z)
        return quantile
    
    def get_people_above_quantile(self, quantile):
        """
        Возвращает порог зарплаты и количество людей выше этого порога
        для заданного квантиля
        
        Args:
            quantile: Значение квантиля (от 0 до 1)
        
        Returns:
            tuple: (порог зарплаты, количество людей выше порога, процент)
        """
        threshold = self.quantile_to_salary(quantile)
        
        if self.population_size is None:
            return threshold, None, (1 - quantile) * 100
        
        count_above = self.population_size * (1 - quantile)
        return threshold, count_above, (1 - quantile) * 100
    
    def get_people_above_salary(self, salary):
        """
        Возвращает квантиль и количество людей выше заданной зарплаты
        
        Args:
            salary: Значение зарплаты
        
        Returns:
            tuple: (квантиль, количество людей ниже, количество людей выше, процент выше)
        """
        quantile = self.salary_to_quantile(salary)
        
        if self.population_size is None:
            return quantile, None, None, (1 - quantile) * 100
        
        count_below = self.population_size * quantile
        count_above = self.population_size - count_below
        return quantile, count_below, count_above, (1 - quantile) * 100