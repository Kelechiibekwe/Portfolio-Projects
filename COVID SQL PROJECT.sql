
/*

Covid 19 Data Exploration 

Skills used: Joins, CTE's, Temp Tables, Windows Functions, Aggregate Functions, Creating Views, Converting Data Types

*/


SELECT * 
FROM Portofolio..['COVID Deaths']
WHERE continent is not null 
ORDER by 3,4

-- Select Data that we are going to be starting with

SELECT Location, date, total_cases, new_cases, total_deaths, population 
FROM Portofolio..['COVID Deaths']
WHERE continent is not null 
ORDER by 1,2

-- Looking at the Total Cases vs Total Deaths

SELECT Location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage 
FROM Portofolio..['COVID Deaths']
Where location like 'Canada' and continent is not null 
ORDER by 1,2

-- Looking at the Total Cases vs Population

SELECT Location, date, population, total_cases,  (total_deaths/population)*100 as PercentofPopulationInfected 
FROM Portofolio..['COVID Deaths']
Where location like 'Canada' and continent is not null 
ORDER by 1,2

-- Looking at Countries with the Highest Infection Rate compared to Population

SELECT Location, Population, MAX(total_cases) as HighestInfectionCount, (MAX(total_cases)/population)*100 as PercentofPopulationInfected 
FROM Portofolio..['COVID Deaths']
--Where location like 'Canada'
GROUP BY Location, Population
ORDER BY PercentofPopulationInfected DESC


-- This is showing the countries with Highest Death Count per Population

SELECT Location, MAX(cast(total_deaths as int)) as TotalDeathCount
FROM Portofolio..['COVID Deaths']
--Where location like 'Canada'
WHERE continent is not null
GROUP BY Location
ORDER BY TotalDeathCount DESC


-- LET'S BREAK THINGS DOWN BY CONTINENT

-- Showing the continent with highest death count per population

SELECT continent, MAX(cast(total_deaths as int)) as TotalDeathCount
FROM Portofolio..['COVID Deaths']
--Where location like 'Canada'
WHERE continent is not null
GROUP BY continent
ORDER BY TotalDeathCount DESC


-- GLOBAL NUMBERS

SELECT date,SUM(new_cases) as TotalCases, SUM(cast(new_deaths as int)) as TotalDeaths, SUM(cast(new_deaths as int))/SUM(new_cases)*100 as DeathPercentage 
FROM Portofolio..['COVID Deaths']
-- Where location like 'Canada' 
WHERE continent is not null 
GROUP by date
ORDER by 1,2

-- Looking at Total Population vs Vaccinations

Select dea.continent,dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(CONVERT(BIGINT,vac.new_vaccinations)) OVER (Partition by dea.location Order by dea.location, dea.date) as RollingPeopleVaccinated
FROM Portofolio..['COVID Deaths'] dea
Join Portofolio..['COVID Vaccination'] vac
	On dea.location = vac.location
	and dea.date = vac.date
WHERE dea.continent is not null
ORDER by 2,3


-- CTE

WITH PopvsVac (Continent, Location, Date, Population, new_vaccinations, RollingPeopleVaccinated)
as
(
Select dea.continent,dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(CONVERT(BIGINT,vac.new_vaccinations)) OVER (Partition by dea.location Order by dea.location, dea.date) as RollingPeopleVaccinated
FROM Portofolio..['COVID Deaths'] dea
Join Portofolio..['COVID Vaccination'] vac
	On dea.location = vac.location
	and dea.date = vac.date
WHERE dea.continent is not null
--ORDER by 2,3
)
SELECT *, (RollingPeopleVaccinated/Population)*100
FROM PopvsVac

--TEMP TABLE
DROP TABLE IF exists #PercentPopulationVaccinated
CREATE TABLE #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar (255),
Date datetime,
Population numeric,
New_vaccinations numeric,
RollingPeopleVaccinated numeric
)

INSERT INTO #PercentPopulationVaccinated
SELECT dea.continent,dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(CONVERT(BIGINT,vac.new_vaccinations)) OVER (Partition by dea.location Order by dea.location, dea.date) as RollingPeopleVaccinated
FROM Portofolio..['COVID Deaths'] dea
Join Portofolio..['COVID Vaccination'] vac
	On dea.location = vac.location
	and dea.date = vac.date
WHERE dea.continent is not null
--ORDER by 2,3

SELECT *, (RollingPeopleVaccinated/Population)*100
FROM #PercentPopulationVaccinated


--Creating View to store data for later visualization

CREATE VIEW PercentPopulationVaccinated as 
SELECT dea.continent,dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(CONVERT(BIGINT,vac.new_vaccinations)) OVER (Partition by dea.location Order by dea.location, dea.date) as RollingPeopleVaccinated
FROM Portofolio..['COVID Deaths'] dea
Join Portofolio..['COVID Vaccination'] vac
	On dea.location = vac.location
	and dea.date = vac.date
WHERE dea.continent is not null
--ORDER by 2,3