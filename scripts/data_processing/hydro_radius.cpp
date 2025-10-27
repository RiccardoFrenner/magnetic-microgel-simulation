#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using Vec3 = std::array<double, 3>;

inline double dist(Vec3 const &a, Vec3 const &b) {
  double dist_squared = 0.0;
  for (std::size_t dim = 0; dim < 3; ++dim) {
    dist_squared += std::pow(b[dim] - a[dim], 2);
  }
  return std::sqrt(dist_squared);
}

double compute_hydrodynamic_radius(std::vector<Vec3> const &points) {
  std::size_t const N = points.size();

  double rh_inv = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = i + 1; j < N; ++j) {
      rh_inv += 1.0 / dist(points[i], points[j]);
    }
  }

  rh_inv *= 2.0 / (N * (N - 1));

  return 1.0 / rh_inv;
}

std::vector<Vec3> load_npy_file(FILE *fp) {

  // Something is wrong with this code --------

  // fseek(fp, 6, SEEK_CUR); // magic string
  // fseek(fp, 1, SEEK_CUR); // major version number of the file format
  // fseek(fp, 1, SEEK_CUR); // minor version number of the file format
  // unsigned char buff[8];
  // fread(buff, sizeof(buff[0]), 8, fp);
  // std::cout << std::hex << (int)buff[0] << (int)buff[1] << (int)buff[2]
  //           << (int)buff[3] << (int)buff[4] << (int)buff[5] << (int)buff[6]
  //           << (int)buff[7] << "\n";

  // // Skip header
  // unsigned char a, b;
  // fread(&a, sizeof(a), 1, fp);
  // fread(&b, sizeof(b), 1, fp);
  // std::cout << "Header: " << std::hex << int(a) << " " << std::hex << int(b)
  //           << "\n";
  // std::uint16_t const header_len = b << 8 | a; // little endian
  // std::cout << "Header length: " << header_len << "\n";
  // fseek(fp, header_len, SEEK_CUR);

  // -------- therefore just skip to the first newline
  {
    char buf[65535 * 2]; // npy header will is smaller than this in version 1.0
    auto const ret = fgets(buf, sizeof(buf), fp);
    if (ret != NULL) {
      // printf("\"%s\"\n", buf);
    } else {
      std::cerr << "Error reading header\n";
    }
  }

  if (ferror(fp)) {
    std::cerr << "ferror: Something is wrong with the header\n";
    exit(EXIT_FAILURE);
  }
  if (feof(fp)) {
    std::cerr << "feof: Something is wrong with the header\n";
    exit(EXIT_FAILURE);
  }

  // Read content
  std::vector<Vec3> points;
  Vec3 buffer;
  points.reserve(650 * 100);
  while (true) {
    auto const ret = fread(&buffer, sizeof(buffer), 1, fp);
    // std::cout << "Ret: " << ret << "\n";
    if (ret != 1) {
      if (ferror(fp)) {
        std::cerr << "Error ferror: ";
      } else if (feof(fp)) {
        break;
      }
      std::cerr << "Array probably does not have dimension (n,3)\n";
      exit(EXIT_FAILURE);
    }

    // std::cout << buffer[0] << ", " << buffer[1] << ", " << buffer[2] << "\n";
    points.push_back(buffer);
  }

  return points;
}

std::vector<Vec3> load_npy_file(std::string const &path_to_file) {
  FILE *fp = fopen(path_to_file.c_str(), "rb");
  auto const result = load_npy_file(fp);
  fclose(fp);
  return result;
}

std::vector<Vec3> load_pos_from_npz(std::string const &path_to_file) {
  std::string const command = "unzip -p " + path_to_file + " pos_folded.npy";
  // std::cout << command << "\n";
  FILE *pipe = popen(command.c_str(), "r");
  if (pipe == NULL) {
    std::cerr << "Error: opening file\n";
    exit(EXIT_FAILURE);
  }

  auto const result = load_npy_file(pipe);
  pclose(pipe);
  return result;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Too few arguments passed\n";
    return 1;
  }

  std::string const program_name = *(argv++);
  std::string const file_name = *(argv++);
  if (file_name.ends_with(".npz")) {
    auto const points = load_pos_from_npz(file_name);
    printf("%s,%g\n", file_name.c_str(), compute_hydrodynamic_radius(points));
  } else if (file_name.ends_with(".npy")) {
    FILE *file = fopen(file_name.c_str(), "rb");
    auto const points = load_npy_file(file);
    printf("%s,%g\n", file_name.c_str(), compute_hydrodynamic_radius(points));
  } else {
    std::cerr << "Invalid file format: '" + file_name + "'\n";
    exit(EXIT_FAILURE);
  }

  // std::cout << compute_hydrodynamic_radius(points) << "\n";
}
