# Filename: login.feature

Feature: User Login
  As a registered user
  I want to log into the application
  So that I can access my personal dashboard

  Background:
    Given the application is running
    And the user database is available

  Scenario: Successful login with valid credentials
    Given the user navigates to the login page
    When the user enters username "testuser" and password "P@ssw0rd"
    And clicks the login button
    Then the user should be redirected to the dashboard
    And the welcome message "Welcome, testuser!" should be displayed

  Scenario: Failed login with invalid password
    Given the user navigates to the login page
    When the user enters username "testuser" and password "wrongpassword"
    And clicks the login button
    Then the user should see an error message "Invalid username or password"

  Scenario Outline: Login with multiple invalid credentials
    Given the user navigates to the login page
    When the user enters username "<username>" and password "<password>"
    And clicks the login button
    Then the user should see an error message "<error_message>"

    Examples:
      | username   | password      | error_message                  |
      | testuser   | 12345         | Invalid username or password   |
      | unknown    | P@ssw0rd      | Invalid username or password   |
      | blank      | blank         | Username and password required |
