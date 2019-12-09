// Copyright 2019 Korobeinikov Aleksei
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <math.h>
#include <utility>
#include <vector>
#include "./calculation_of_integrals.h"

double func1(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    return (x * x - 2 * y);
}

double func2(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    double z = v[2];
    return (x + y*y + z*z*z);
}

double func3(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    double z = v[2];
    return (log10(2*x*x) + sqrt(z) + 5*y);
}

double func4(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    double z = v[2];
    return (exp(x) - sqrt(10) * 5 * sin(y) + cos(-2 * z * z));
}

double func5(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    double z = v[2];
    double t = v[3];
    return x + y +  z +  t;
}

double func6(std::vector<double> v) {
    double x = v[0];
    double y = v[1];
    double z = v[2];
    double t = v[3];
    return cos(5*x) + exp(y) + 2.9*sin(z) - t*t;
}

TEST(MultipleIntegraion, Integral_with_2_dimension) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 2;
    std::vector<std::pair<double, double>> dist(n);
    if (rank == 0) {
        dist[0] = { 4, 10 };
        dist[1] = { 1, 56 };
    }

    double result = ParallelVersion(func1, dist);

    if (rank == 0) {
        double error = 0.1;
        ASSERT_NEAR(result, -1650.0990000000018, error);
    }
}

TEST(MultipleIntegraion, Integral_with_3_dimension) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 3;
    std::vector<std::pair<double, double>> dist(n);
    if (rank == 0) {
        dist[0] = { 0, 1 };
        dist[1] = { -13, 5 };
        dist[2] = { 3, 7 };
    }

    double result = ParallelVersion(func2, dist);

    if (rank == 0) {
        double error = 0.1;
        ASSERT_NEAR(result, 13571.661599999945, error);
    }
}

TEST(MultipleIntegraion, Integral_with_3_dimension_and_use_log_function) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 3;
    std::vector<std::pair<double, double>> dist(n);
    if (rank == 0) {
        dist[0] = { 0, 1 };
        dist[1] = { -13, 5 };
        dist[2] = { 3, 7 };
    }

    double result = ParallelVersion(func3, dist);

    if (rank == 0) {
        double error = 0.1;
        ASSERT_NEAR(result, -1320.7583639973182, error);
    }
}

TEST(MultipleIntegraion, Integral_with_3_dimension_and_use_sin_and_cos_functions) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 3;
    std::vector<std::pair<double, double>> dist(n);
    if (rank == 0) {
        dist[0] = { -9, 1 };
        dist[1] = { -100, 100 };
        dist[2] = { -2, 2 };
    }

    double result = ParallelVersion(func4, dist);

    if (rank == 0) {
        double error = 0.01;
        ASSERT_NEAR(result, 4442.00198999875, error);
    }
}

TEST(MultipleIntegraion, First_Integral_with_4_dimension_easy_version) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 4;
    std::vector<std::pair<double, double>> dist(n);
    if (rank == 0) {
        dist[0] = { -2, 2 };
        dist[1] = { -3.3, 5 };
        dist[2] = { -9, 1 };
        dist[3] = { 5, 7 };
    }
    double time1 = MPI_Wtime();
    double result = ParallelVersion(func5, dist);
    double time2 = MPI_Wtime();
    if (rank == 0) {
        std::cout << time2 - time1 << '\n';
        double error = 0.0001;
        ASSERT_NEAR(result, 1892.4, error);
    }
}

TEST(MultipleIntegraion, Second_Integral_with_4_dimension_hard_version) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 4;
    std::vector<std::pair<double, double>> dist(n);
    if (rank == 0) {
        dist[0] = { -4.5, 0 };
        dist[1] = { 0, 5 };
        dist[2] = { -58, 12 };
        dist[3] = { 5, 73 };
    }
    double time1 = MPI_Wtime();
    double result = ParallelVersion(func6, dist);
    double time2 = MPI_Wtime();
    if (rank == 0) {
        double error = 0.01;
        std::cout << time2 - time1 << '\n';
        // std::cout << std::fixed << result;
        ASSERT_NEAR(result, -201012517.928, error);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
