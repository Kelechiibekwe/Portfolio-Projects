/*

Cleaning Data in SQL Queries

*/

SELECT *
FROM Portofolio..NashvilleHousing

--------------------------------------------------------------------------------------------------------------------------

--Standardize Date Format

SELECT SaleDateConverted, CONVERT(Date, SaleDate)
FROM Portofolio..NashvilleHousing

ALTER TABLE NashvilleHousing
Add SaleDateConverted Date;

UPDATE NashvilleHousing 
SET SaleDateConverted = CONVERT(Date,SaleDate)


--------------------------------------------------------------------------------------------------------------------------

--Populated Property Address data

SELECT 
FROM Portofolio..NashvilleHousing a
JOIN Portofolio..NashvilleHousing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
