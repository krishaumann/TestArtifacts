Feature: Profile Management

  Scenario: Update phone number
    Given the user is on the profile page
    When the user updates phone number
    Then the change is saved

  Scenario: Change password
    Given the user is on the profile page
    When the user changes password
    Then the password is updated

  Scenario: Request password reset
    Given the user is on the profile page
    When the user requests password reset
    Then a reset email is sent