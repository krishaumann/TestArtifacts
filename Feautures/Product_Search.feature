Feature: Product Search

  Scenario: Search for existing product
    Given the user searches for product 'Laptop'
    Then search results show laptops

  Scenario: Search for non-existing product
    Given the user searches for a non-existing product
    Then no results are found

  Scenario: Filter by brand
    Given the user filters by brand
    Then only matching brand laptops are displayed

  Scenario: Sort by price
    Given the user sorts by price ascending
    Then products are ordered by price
