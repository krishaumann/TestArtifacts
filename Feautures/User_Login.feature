Feature: User Login

  Scenario: Login with valid credentials
    Given the user is on the login page
    When the user enters valid username and password
    And clicks login
    Then the user is redirected to dashboard

  Scenario: Login with invalid credentials
    Given the user is on the login page
    When the user enters invalid credentials
    Then an error message is displayed

  Scenario: Logout
    Given the user is logged in
    When the user clicks logout
    Then the user is redirected to login page