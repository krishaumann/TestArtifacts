Feature: Checkout Process

  Scenario: Proceed to checkout with valid card
    Given the user adds product to cart
    When the user proceeds to checkout
    And enters shipping address
    And retries with valid card
    Then payment is successful

  Scenario: Proceed to checkout with invalid card
    Given the user adds product to cart
    When the user proceeds to checkout
    And enters invalid card number
    Then an error message is displayed