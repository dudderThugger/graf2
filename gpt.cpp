#include <iostream>
#include <cmath>

using namespace std;

const double g = 9.8;  // Acceleration due to gravity (in m/s^2)

double calculateOrientation(double springConstant, double time, double height, double mass, double length)
{
    double angularVelocity = sqrt((g + (springConstant / mass) * height) / length);
    return angularVelocity * time;
}

int main()
{
    double springConstant, time, height, mass, length;

    cout << "Enter the spring constant (c): ";
    cin >> springConstant;

    cout << "Enter the time (t): ";
    cin >> time;

    cout << "Enter the height (h): ";
    cin >> height;

    cout << "Enter the mass (m): ";
    cin >> mass;

    cout << "Enter the length (L): ";
    cin >> length;

    double orientation = calculateOrientation(springConstant, time, height, mass, length);

    cout << "Cuboid's orientation: " << orientation << " rad" << endl;

    return 0;
}
