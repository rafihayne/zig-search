const std = @import("std");
const graph = @import("graph.zig");
const stdout = std.io.getStdOut().outStream();




pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = &arena.allocator;
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer {
    //     const leaked = gpa.deinit();
    //     // if (leaked) std.debug.warn("Leaking mem\n", .{}); //fail test
    // }
    // var allocator = &gpa.allocator;

    var g = graph.Graph(graph.Vec2, f32).init(allocator);
    defer g.deinit();


    // Parse dataset
    // From: https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm
    const mb = 1000000; 

    // Block to reuse variable names. seems ugly too
    _ = addnodes: {
        const node_path = "./data/cal.cnode";
        const node_data = try std.fs.cwd().readFileAlloc(allocator, node_path, mb);
        defer allocator.free(node_data);
        var lines = std.mem.tokenize(node_data, "\n");
    
        // node_id long lat
        while (lines.next()) |line| {
            var tokens = std.mem.tokenize(line, " ");
            // First token is the index we don't care about
            _ = tokens.next();

            // this is soooo ugly yikes
            var temp = [2]f32{0.0, 0.0};
            var curr:u2 = 0;
            while(tokens.next()) |token| {
                const value = try std.fmt.parseFloat(f32, token);
                if (curr == 0) {
                    temp[0] = value;
                } else {
                    temp[1] = value;
                }
                curr += 1;
            }
            // dataset is in long, lat
            _ = try g.addNode(graph.Vec2{ .x = temp[1], .y = temp[0] });
        }
        break :addnodes;
    };

    _ = addedges: {
        const edge_path = "./data/cal.cedge";
        const edge_data = try std.fs.cwd().readFileAlloc(allocator, edge_path, mb);
        defer allocator.free(edge_data);
        var lines = std.mem.tokenize(edge_data, "\n");
    
        // edge_id node_start_id node_end_id l2_dist
        while (lines.next()) |line| {
            var tokens = std.mem.tokenize(line, " ");
            // First token is the index we don't care about
            _ = tokens.next();

            // this is soooo ugly yikes
            var temp_edge = [2]usize{0, 0};
            var temp_dist: f32 = 0;
            var curr:u4 = 0;
            while(tokens.next()) |token| {
                if (curr == 0) {
                    const value = try std.fmt.parseInt(usize, token, 10);
                    temp_edge[0] = value;
                } else if (curr == 1){
                    const value = try std.fmt.parseInt(usize, token, 10);
                    temp_edge[1] = value;
                } else {
                    const value = try std.fmt.parseFloat(f32, token);
                    temp_dist = value;
                }
                curr += 1;
            }
            _ = try g.addEdgeBidirectional(temp_edge[0], temp_edge[1], temp_dist);
        }
        break :addedges;
    };

    var timer = try std.time.Timer.start();
    // approx sacramento to sandiego
    var result = try g.AStarSearch(7261, 20286, graph.euclidean);
    try stdout.print("{} nanoseconds elapsed\n", .{timer.read()});
    defer result.deinit();
    try result.print();
}