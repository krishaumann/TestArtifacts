Feature: User Registration

  Scenario: Successful registration and email verification
    Given the user opens the registration form
    When the user fills in details
    And submits registration
    Then the account is activated

  Scenario: Registration with missing details
    Given the user opens the registration form
    When the user leaves fields empty
    Then an error message is displayed